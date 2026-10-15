// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineTokenizer.h"

namespace PipelineTokenizerDecodeSpecPrivate
{
	struct FSttClip
	{
		FString Id;
		TArray<int32> TokenIds;
		FString Text;
		TArray<FString> Words; // word_frames[*][0]
	};

	static constexpr int32 BlankId = 8192;
	static const TCHAR WordMark = (TCHAR)0x2581; // U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece word start

	static const TCHAR ReplacementChar = (TCHAR)0xFFFD;

	static FString GetDevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		if (!Env.IsEmpty())
		{
			return Env;
		}
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev"));
	}

	static const TCHAR* FetchHint()
	{
		return TEXT("To create it: run D:/Micromamba/micromambaenv/envs/pipelines/python.exe ModelPipelines-UE/Tools/make_stt_goldens.py "
			"and the STT model fetch (phonon-2-onnx) so <dev>/models/stt/phonon-2-onnx/fp16w/tokenizer.json exists.");
	}

	static bool LoadTokenizerJson(FAutomationTestBase& Test, FString& OutJson)
	{
		const FString Path = GetDevDir() / TEXT("models/stt/phonon-2-onnx/fp16w/tokenizer.json");
		if (!FFileHelper::LoadFileToString(OutJson, *Path))
		{
			Test.AddError(FString::Printf(TEXT("Missing Phonon tokenizer '%s'. %s"), *Path, FetchHint()));
			return false;
		}
		return true;
	}

	static bool LoadClips(FAutomationTestBase& Test, TArray<FSttClip>& Out)
	{
		const FString Path = GetDevDir() / TEXT("goldens/stt/stt_goldens.json");
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path))
		{
			Test.AddError(FString::Printf(TEXT("Missing STT goldens '%s'. %s"), *Path, FetchHint()));
			return false;
		}
		TSharedPtr<FJsonObject> Root;
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Root) || !Root.IsValid())
		{
			Test.AddError(FString::Printf(TEXT("STT goldens '%s' is not valid JSON. %s"), *Path, FetchHint()));
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>>* ClipsArr = nullptr;
		if (!Root->TryGetArrayField(TEXT("clips"), ClipsArr))
		{
			Test.AddError(FString::Printf(TEXT("STT goldens '%s' has no 'clips' array"), *Path));
			return false;
		}
		for (const TSharedPtr<FJsonValue>& V : *ClipsArr)
		{
			const TSharedPtr<FJsonObject> O = V->AsObject();
			if (!O.IsValid()) { continue; }
			const TSharedPtr<FJsonObject>* Onnx = nullptr;
			if (!O->TryGetObjectField(TEXT("onnx"), Onnx)) { continue; } // e.g. short_319: onnx is null
			FSttClip C;
			C.Id = O->GetStringField(TEXT("id"));
			(*Onnx)->TryGetStringField(TEXT("text"), C.Text);
			const TArray<TSharedPtr<FJsonValue>>* Ids = nullptr;
			if ((*Onnx)->TryGetArrayField(TEXT("token_ids"), Ids))
			{
				for (const TSharedPtr<FJsonValue>& I : *Ids) { C.TokenIds.Add((int32)I->AsNumber()); }
			}
			const TArray<TSharedPtr<FJsonValue>>* WF = nullptr;
			if ((*Onnx)->TryGetArrayField(TEXT("word_frames"), WF))
			{
				for (const TSharedPtr<FJsonValue>& W : *WF)
				{
					const TArray<TSharedPtr<FJsonValue>>& Row = W->AsArray();
					C.Words.Add(Row.Num() > 0 ? Row[0]->AsString() : FString());
				}
			}
			Out.Add(MoveTemp(C));
		}
		if (Out.Num() < 40)
		{
			Test.AddError(FString::Printf(TEXT("STT goldens '%s' has only %d clips with an onnx block, expected at least 40. %s"), *Path, Out.Num(), FetchHint()));
			return false;
		}
		return true;
	}

	static TStrongObjectPtr<UPipelineTokenizer> MakeTok(FAutomationTestBase& Test, const FString& Json)
	{
		TStrongObjectPtr<UPipelineTokenizer> Tok(NewObject<UPipelineTokenizer>(GetTransientPackage()));
		if (!Tok->InitializeTokenizerFromJson(Json))
		{
			Test.AddError(FString::Printf(TEXT("InitializeTokenizerFromJson failed on the Phonon tokenizer: %s"), *Tok->GetLastError()));
		}
		return Tok;
	}

	static FString Trim(const FString& S) { return S.TrimStartAndEnd(); }

	static FString IdsStr(TConstArrayView<int32> Ids)
	{
		FString S;
		for (int32 i = 0; i < Ids.Num() && i < 16; ++i) { S += FString::Printf(TEXT("%d "), Ids[i]); }
		return S + (Ids.Num() > 16 ? TEXT("...") : TEXT(""));
	}

	/** Expect exactly one Warning from the named function (message text is "UPipelineTokenizer::<Fn>: <reason>"; the reason is not specified). */
	static void ExpectOneWarning(FAutomationTestBase& Test, const TCHAR* Fn)
	{
		Test.AddExpectedMessagePlain(FString::Printf(TEXT("UPipelineTokenizer::%s:"), Fn), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
	}

	static bool LastErrorNames(FAutomationTestBase& Test, UPipelineTokenizer& Tok, const TCHAR* Fn, const TCHAR* What)
	{
		const FString Prefix = FString::Printf(TEXT("UPipelineTokenizer::%s: "), Fn);
		const FString Err = Tok.GetLastError();
		const bool bOk = Err.StartsWith(Prefix) && Err.Len() > Prefix.Len();
		Test.TestTrue(*FString::Printf(TEXT("%s: GetLastError expected to start with '%s' and give a reason, got '%s'"), What, *Prefix, *Err), bOk);
		return bOk;
	}
}

using namespace PipelineTokenizerDecodeSpecPrivate;

// Lead (2026-10-15): FString ==/!= and TestEqual(FString) are case-INSENSITIVE; parity must be exact.
static bool SameCaseSensitive(const TArray<FString>& A, const TArray<FString>& B)
{
	if (A.Num() != B.Num()) { return false; }
	for (int32 i = 0; i < A.Num(); ++i) { if (!A[i].Equals(B[i], ESearchCase::CaseSensitive)) { return false; } }
	return true;
}

BEGIN_DEFINE_SPEC(FPipelineTokenizerDecodeSpec, "Pipelines.Tokenizer.Decode",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
END_DEFINE_SPEC(FPipelineTokenizerDecodeSpec)

void FPipelineTokenizerDecodeSpec::Define()
{
	Describe("Parity", [this]()
	{
		It("Decode(onnx.token_ids, skip special) trimmed equals onnx.text on every golden clip (acceptance 1)", [this]()
		{
			FString Json; TArray<FSttClip> Clips;
			if (!LoadTokenizerJson(*this, Json) || !LoadClips(*this, Clips)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTok(*this, Json);
			int32 Bad = 0;
			for (const FSttClip& C : Clips)
			{
				FString Got = TEXT("garbage"); // must be overwritten, not appended to
				const bool bOk = Tok->Decode(C.TokenIds, true, Got);
				if (!bOk || !Trim(Got).Equals(C.Text, ESearchCase::CaseSensitive))
				{
					++Bad;
					AddError(FString::Printf(TEXT("clip %s: Decode ok=%d (err '%s') expected '%s' got '%s' (ids: %s)"),
						*C.Id, bOk, *Tok->GetLastError(), *C.Text, *Trim(Got), *IdsStr(C.TokenIds)));
				}
			}
			TestEqual(*FString::Printf(TEXT("clips whose decoded text differs from golden, of %d (expected 0)"), Clips.Num()), Bad, 0);
		});

		It("words rebuilt from IdToToken pieces (new word at U+2581) decode to word_frames words (acceptance 2)", [this]()
		{
			FString Json; TArray<FSttClip> Clips;
			if (!LoadTokenizerJson(*this, Json) || !LoadClips(*this, Clips)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTok(*this, Json);
			int32 Bad = 0;
			for (const FSttClip& C : Clips)
			{
				TArray<TArray<int32>> Groups;
				bool bPieceFail = false;
				for (int32 Id : C.TokenIds)
				{
					FString Piece;
					if (!Tok->IdToToken(Id, Piece)) { bPieceFail = true; AddError(FString::Printf(TEXT("clip %s: IdToToken(%d) failed: %s"), *C.Id, Id, *Tok->GetLastError())); break; }
					if (Groups.Num() == 0 || Piece.StartsWith(FString(1, &WordMark)))
					{
						Groups.AddDefaulted();
					}
					Groups.Last().Add(Id);
				}
				TArray<FString> Words;
				for (const TArray<int32>& G : Groups)
				{
					FString W;
					Tok->Decode(G, true, W);
					Words.Add(Trim(W));
				}
				if (bPieceFail || !SameCaseSensitive(Words, C.Words))
				{
					++Bad;
					AddError(FString::Printf(TEXT("clip %s: words expected [%s] got [%s]"), *C.Id, *FString::Join(C.Words, TEXT("|")), *FString::Join(Words, TEXT("|"))));
				}
			}
			TestEqual(*FString::Printf(TEXT("clips whose rebuilt words differ, of %d (expected 0)"), Clips.Num()), Bad, 0);
		});

		It("a long Decode followed by a short one leaves no stale text (handle buffer copied, not aliased)", [this]()
		{
			FString Json; TArray<FSttClip> Clips;
			if (!LoadTokenizerJson(*this, Json) || !LoadClips(*this, Clips)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTok(*this, Json);
			const FSttClip* Long = nullptr; const FSttClip* Short = nullptr;
			for (const FSttClip& C : Clips)
			{
				if (C.TokenIds.Num() == 0) { continue; }
				if (!Long || C.TokenIds.Num() > Long->TokenIds.Num()) { Long = &C; }
				if (!Short || C.TokenIds.Num() < Short->TokenIds.Num()) { Short = &C; }
			}
			if (!Long || !Short) { AddError(TEXT("no clip with tokens in goldens")); return; }
			FString A, B, A2;
			TestTrue(TEXT("long Decode ok"), Tok->Decode(Long->TokenIds, true, A));
			TestTrue(TEXT("short Decode ok"), Tok->Decode(Short->TokenIds, true, B));
			// A kept the first result: it must still equal the golden after the second call reused the handle buffer.
			TestTrue(*FString::Printf(TEXT("long clip %s text kept after next call"), *Long->Id), Trim(A).Equals(Long->Text, ESearchCase::CaseSensitive));
			TestTrue(*FString::Printf(TEXT("short clip %s text exact (no tail of the long one)"), *Short->Id), Trim(B).Equals(Short->Text, ESearchCase::CaseSensitive));
			TestTrue(TEXT("long Decode again ok"), Tok->Decode(Long->TokenIds, true, A2));
			TestEqual(TEXT("repeat Decode deterministic"), A2, A);
		});
	});

	Describe("SpecialTokens", [this]()
	{
		It("IdToToken(8192) is <blank>; Decode({8192}) is empty when skipping and contains <blank> otherwise (acceptance 3)", [this]()
		{
			FString Json;
			if (!LoadTokenizerJson(*this, Json)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTok(*this, Json);
			FString Piece = TEXT("x");
			TestTrue(TEXT("IdToToken(8192) expected true"), Tok->IdToToken(BlankId, Piece));
			TestTrue(TEXT("IdToToken(8192) expected '<blank>'"), Piece.Equals(TEXT("<blank>"), ESearchCase::CaseSensitive));
			const TArray<int32> Blank = { BlankId };
			FString Skipped = TEXT("x"), Kept;
			TestTrue(TEXT("Decode({8192}, true) expected true"), Tok->Decode(Blank, true, Skipped));
			TestEqual(TEXT("Decode({8192}, true) expected empty (special token skipped)"), Skipped, FString());
			TestTrue(TEXT("Decode({8192}, false) expected true"), Tok->Decode(Blank, false, Kept));
			TestTrue(*FString::Printf(TEXT("Decode({8192}, false) expected to contain <blank>, got '%s'"), *Kept), Kept.Contains(TEXT("<blank>")));
		});

		It("a word-start piece keeps its U+2581 (non-ASCII piece survives UTF-8 to FString with explicit length)", [this]()
		{
			FString Json; TArray<FSttClip> Clips;
			if (!LoadTokenizerJson(*this, Json) || !LoadClips(*this, Clips)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTok(*this, Json);
			int32 Starts = 0, Bad = 0;
			for (const FSttClip& C : Clips)
			{
				for (int32 Id : C.TokenIds)
				{
					FString Piece;
					if (!Tok->IdToToken(Id, Piece)) { ++Bad; continue; }
					// A NUL-terminated read of a non-terminated buffer, or a Latin-1 conversion, would garble or lengthen this.
					if (Piece.StartsWith(FString(1, &WordMark))) { ++Starts; }
					if (Piece.IsEmpty() || Piece.Contains(FString(1, &ReplacementChar))) { ++Bad; }
				}
			}
			TestTrue(*FString::Printf(TEXT("pieces beginning with U+2581 across all clips expected > 0, got %d"), Starts), Starts > 0);
			TestEqual(TEXT("failed, empty or U+FFFD pieces for emitted ids (expected 0)"), Bad, 0);
		});
	});

	Describe("NonAscii", [this]()
	{
		It("Decode(Encode(\"Zoe cafe - naive\" with diacritics and em dash)) matches the Python reference (acceptance 4; em dash is <unk>)", [this]()
		{
			FString Json;
			if (!LoadTokenizerJson(*this, Json)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTok(*this, Json);
			// Zoë café — naïve
			FString Text;
			Text += TEXT("Zo"); Text.AppendChar((TCHAR)0x00EB); Text += TEXT(" caf"); Text.AppendChar((TCHAR)0x00E9);
			Text += TEXT(" "); Text.AppendChar((TCHAR)0x2014); Text += TEXT(" na"); Text.AppendChar((TCHAR)0x00EF); Text += TEXT("ve");
			const TArray<int32> Ids = Tok->Encode(Text, false);
			TestTrue(*FString::Printf(TEXT("Encode produced ids (error '%s')"), *Tok->GetLastError()), Ids.Num() > 0);
			FString Out;
			TestTrue(TEXT("Decode expected true"), Tok->Decode(Ids, true, Out));
			// Lead (2026-10-15): the Phonon vocabulary has no em dash -> <unk> (id 0), which decode skips. Expected values are
			// the Python reference (tokenizers 0.23.3 on the same tokenizer.json), not a round trip:
			//   ids [681, 7869, 8101, 298, 1389, 7906, 7863, 0, 358, 8129, 507], decode(skip_special) = "Zoë café  naïve".
			const TArray<int32> WantIds = { 681, 7869, 8101, 298, 1389, 7906, 7863, 0, 358, 8129, 507 };
			TestTrue(*FString::Printf(TEXT("ids equal the Python reference (got %s)"), *IdsStr(Ids)), Ids == WantIds);
			FString Want;
			Want += TEXT("Zo"); Want.AppendChar((TCHAR)0x00EB); Want += TEXT(" caf"); Want.AppendChar((TCHAR)0x00E9);
			Want += TEXT("  na"); Want.AppendChar((TCHAR)0x00EF); Want += TEXT("ve");
			TestTrue(*FString::Printf(TEXT("decode expected '%s' got '%s' (case-sensitive)"), *Want, *Trim(Out)),
				Trim(Out).Equals(Want, ESearchCase::CaseSensitive));
		});
	});

	Describe("Errors", [this]()
	{
		It("uninitialized: Decode and IdToToken return false, empty output, one Warning naming the function each (acceptance 5)", [this]()
		{
			TStrongObjectPtr<UPipelineTokenizer> Tok(NewObject<UPipelineTokenizer>(GetTransientPackage()));
			ExpectOneWarning(*this, TEXT("Decode"));
			FString Text = TEXT("stale");
			const TArray<int32> Ids = { 1, 2, 3 };
			TestFalse(TEXT("Decode uninitialized expected false"), Tok->Decode(Ids, true, Text));
			TestEqual(TEXT("OutText expected empty after failure"), Text, FString());
			LastErrorNames(*this, *Tok, TEXT("Decode"), TEXT("Decode uninitialized"));

			ExpectOneWarning(*this, TEXT("IdToToken"));
			FString Piece = TEXT("stale");
			TestFalse(TEXT("IdToToken uninitialized expected false"), Tok->IdToToken(5, Piece));
			LastErrorNames(*this, *Tok, TEXT("IdToToken"), TEXT("IdToToken uninitialized"));
		});

		It("negative id: Decode and IdToToken return false with one Warning each; the object stays usable (acceptance 5)", [this]()
		{
			FString Json;
			if (!LoadTokenizerJson(*this, Json)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTok(*this, Json);
			ExpectOneWarning(*this, TEXT("Decode"));
			FString Text = TEXT("stale");
			// A cast of -1 to uint32 would reach the library as 4294967295.
			const TArray<int32> Bad = { 10, -1, 20 };
			TestFalse(TEXT("Decode with a negative id expected false"), Tok->Decode(Bad, true, Text));
			TestEqual(TEXT("OutText expected empty after failure"), Text, FString());
			LastErrorNames(*this, *Tok, TEXT("Decode"), TEXT("Decode negative id"));

			ExpectOneWarning(*this, TEXT("IdToToken"));
			FString Piece = TEXT("stale");
			TestFalse(TEXT("IdToToken(-1) expected false"), Tok->IdToToken(-1, Piece));
			LastErrorNames(*this, *Tok, TEXT("IdToToken"), TEXT("IdToToken negative id"));

			FString Good;
			TestTrue(TEXT("IdToToken valid after failures"), Tok->IdToToken(BlankId, Good));
			TestTrue(*FString::Printf(TEXT("GetLastError expected empty after a successful call, got '%s'"), *Tok->GetLastError()), Tok->GetLastError().IsEmpty());
		});

		It("id >= vocab: IdToToken returns true and \"\" with no warning (acceptance 5)", [this]()
		{
			FString Json;
			if (!LoadTokenizerJson(*this, Json)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTok(*this, Json);
			for (int32 Id : { 8193, 100000, 2000000000 })
			{
				FString Piece = TEXT("stale");
				TestTrue(*FString::Printf(TEXT("IdToToken(%d) expected true (unknown id is not an error)"), Id), Tok->IdToToken(Id, Piece));
				TestEqual(*FString::Printf(TEXT("IdToToken(%d) expected empty piece"), Id), Piece, FString());
				TestTrue(*FString::Printf(TEXT("GetLastError expected empty after IdToToken(%d), got '%s'"), Id, *Tok->GetLastError()), Tok->GetLastError().IsEmpty());
			}
		});

		It("empty Ids: Decode returns true and \"\" with no warning, and clears a previous error (acceptance 5)", [this]()
		{
			FString Json;
			if (!LoadTokenizerJson(*this, Json)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTok(*this, Json);
			ExpectOneWarning(*this, TEXT("IdToToken"));
			FString Piece;
			TestFalse(TEXT("IdToToken(-5) expected false"), Tok->IdToToken(-5, Piece));
			TestFalse(TEXT("GetLastError expected non-empty after failure"), Tok->GetLastError().IsEmpty());
			FString Text = TEXT("stale");
			TestTrue(TEXT("Decode({}) expected true"), Tok->Decode(TConstArrayView<int32>(), true, Text));
			TestEqual(TEXT("Decode({}) expected empty text"), Text, FString());
			TestTrue(TEXT("GetLastError expected empty after a successful empty Decode"), Tok->GetLastError().IsEmpty());
			Text = TEXT("stale");
			TestTrue(TEXT("Decode({}, false) expected true"), Tok->Decode(TConstArrayView<int32>(), false, Text));
			TestEqual(TEXT("Decode({}, false) expected empty text"), Text, FString());
		});
	});

	Describe("Threads", [this]()
	{
		It("8 threads call Decode and IdToToken on one tokenizer for 2 s; results equal the serial ones (acceptance 6)", [this]()
		{
			FString Json; TArray<FSttClip> Clips;
			if (!LoadTokenizerJson(*this, Json) || !LoadClips(*this, Clips)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTok(*this, Json);

			// Serial reference computed on this thread first.
			TArray<FString> SerialText;
			TArray<TArray<FString>> SerialPieces;
			for (const FSttClip& C : Clips)
			{
				FString T; Tok->Decode(C.TokenIds, true, T);
				SerialText.Add(T);
				TArray<FString>& P = SerialPieces.AddDefaulted_GetRef();
				for (int32 Id : C.TokenIds) { FString Piece; Tok->IdToToken(Id, Piece); P.Add(Piece); }
			}

			const int32 NumThreads = 8;
			TSharedRef<FThreadSafeCounter> Mismatches = MakeShared<FThreadSafeCounter>();
			TSharedRef<FThreadSafeCounter> Calls = MakeShared<FThreadSafeCounter>();
			UPipelineTokenizer* TokPtr = Tok.Get();
			const TArray<FSttClip>* ClipsPtr = &Clips;
			const TArray<FString>* TextPtr = &SerialText;
			const TArray<TArray<FString>>* PiecesPtr = &SerialPieces;
			TArray<TFuture<void>> Futures;
			for (int32 T = 0; T < NumThreads; ++T)
			{
				Futures.Add(Async(EAsyncExecution::Thread, [TokPtr, ClipsPtr, TextPtr, PiecesPtr, Mismatches, Calls, T]()
				{
					const double End = FPlatformTime::Seconds() + 2.0;
					int32 i = T;
					while (FPlatformTime::Seconds() < End)
					{
						const int32 Ci = i % ClipsPtr->Num();
						const FSttClip& C = (*ClipsPtr)[Ci];
						if ((i & 1) == 0)
						{
							FString Out;
							const bool bOk = TokPtr->Decode(C.TokenIds, true, Out);
							if (!bOk || !Out.Equals((*TextPtr)[Ci], ESearchCase::CaseSensitive)) { Mismatches->Increment(); }
						}
						else
						{
							for (int32 k = 0; k < C.TokenIds.Num(); ++k)
							{
								FString Piece;
								const bool bOk = TokPtr->IdToToken(C.TokenIds[k], Piece);
								if (!bOk || !Piece.Equals((*PiecesPtr)[Ci][k], ESearchCase::CaseSensitive)) { Mismatches->Increment(); }
							}
						}
						Calls->Increment();
						++i;
					}
				}));
			}
			bool bTimedOut = false;
			const double Deadline = FPlatformTime::Seconds() + 60.0;
			for (TFuture<void>& F : Futures)
			{
				const double Remaining = Deadline - FPlatformTime::Seconds();
				if (Remaining <= 0.0 || !F.WaitFor(FTimespan::FromSeconds(Remaining))) { bTimedOut = true; break; }
			}
			TestFalse(TEXT("workers finished within 60 s of the 2 s run (a deadlock would time out)"), bTimedOut);
			if (bTimedOut) { return; } // workers still reference this scope's data: do not tear down further
			TestTrue(*FString::Printf(TEXT("calls completed expected > 0, got %d"), Calls->GetValue()), Calls->GetValue() > 0);
			TestEqual(*FString::Printf(TEXT("results differing from serial across %d calls (expected 0)"), Calls->GetValue()), Mismatches->GetValue(), 0);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
