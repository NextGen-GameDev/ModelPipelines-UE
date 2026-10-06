// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformMisc.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/StructOnScope.h"
#include "UObject/UnrealType.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineTokenizer.h"

namespace PipelineTokenizerSpecPrivate
{
	struct FGoldenCase
	{
		FString Id;
		FString Text;
		TArray<int32> IdsWithSpecial;
		TArray<int32> IdsWithoutSpecial;
	};

	struct FGoldenBatch
	{
		FString Id;
		TArray<FString> CaseIds;
		bool bAddSpecialTokens = false;
		int32 MaxLength = 0; // JSON null -> 0 (no truncation)
		int32 PadId = 0;
		TArray<TArray<int32>> InputIds;
		TArray<TArray<int32>> AttentionMask;
	};

	struct FGolden
	{
		FString Name;
		FString TokenizerJson; // contents of tokenizer.json
		int32 VocabSize = -1;
		TArray<FGoldenCase> Cases;
		TArray<FGoldenBatch> Batches;

		const FGoldenCase* FindCase(const FString& Id) const
		{
			return Cases.FindByPredicate([&Id](const FGoldenCase& C) { return C.Id == Id; });
		}
	};

	static const TCHAR* GoldenNames[] = { TEXT("gpt2"), TEXT("bert-base-uncased"), TEXT("t5-small") };

	static FString GetGoldenDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("TOKENIZERS_GOLDENS_DIR"));
		if (!Env.IsEmpty())
		{
			return Env;
		}
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev/goldens/tokenizers"));
	}

	static TArray<int32> ReadIntArray(const TArray<TSharedPtr<FJsonValue>>& Values)
	{
		TArray<int32> Out;
		for (const TSharedPtr<FJsonValue>& V : Values)
		{
			Out.Add((int32)V->AsNumber());
		}
		return Out;
	}

	/** Loads one golden JSON and its tokenizer.json. Adds an error naming the path and fetch command on failure. */
	static bool LoadGolden(FAutomationTestBase& Test, const FString& Name, FGolden& Out)
	{
		const FString Dir = GetGoldenDir();
		const FString GoldenPath = Dir / (Name + TEXT(".json"));
		const FString Fetch = TEXT("run: D:/Micromamba/micromambaenv/envs/pipelines/python.exe ModelPipelines-UE/Tools/make_tokenizer_goldens.py");
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *GoldenPath))
		{
			Test.AddError(FString::Printf(TEXT("Missing golden '%s'. To create it, %s"), *GoldenPath, *Fetch));
			return false;
		}
		TSharedPtr<FJsonObject> Root;
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Root) || !Root.IsValid())
		{
			Test.AddError(FString::Printf(TEXT("Golden '%s' is not valid JSON. To regenerate, %s"), *GoldenPath, *Fetch));
			return false;
		}
		const TSharedPtr<FJsonObject>* TokObj = nullptr;
		if (!Root->TryGetObjectField(TEXT("tokenizer"), TokObj))
		{
			Test.AddError(FString::Printf(TEXT("Golden '%s' has no 'tokenizer' object"), *GoldenPath));
			return false;
		}
		Out.Name = Name;
		double Vocab = -1;
		if ((*TokObj)->TryGetNumberField(TEXT("vocab_size_with_added"), Vocab))
		{
			Out.VocabSize = (int32)Vocab;
		}
		FString RelFile;
		(*TokObj)->TryGetStringField(TEXT("file"), RelFile);
		const FString TokPath = FPaths::ConvertRelativePathToFull(Dir, RelFile);
		if (RelFile.IsEmpty() || !FFileHelper::LoadFileToString(Out.TokenizerJson, *TokPath))
		{
			Test.AddError(FString::Printf(TEXT("Missing tokenizer.json '%s' (named by golden '%s'). To fetch the tokenizer files and goldens, %s"),
				*TokPath, *GoldenPath, *Fetch));
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>>* CasesArr = nullptr;
		if (!Root->TryGetArrayField(TEXT("cases"), CasesArr))
		{
			Test.AddError(FString::Printf(TEXT("Golden '%s' has no 'cases'"), *GoldenPath));
			return false;
		}
		for (const TSharedPtr<FJsonValue>& V : *CasesArr)
		{
			const TSharedPtr<FJsonObject> O = V->AsObject();
			FGoldenCase C;
			C.Id = O->GetStringField(TEXT("id"));
			C.Text = O->GetStringField(TEXT("text"));
			C.IdsWithSpecial = ReadIntArray(O->GetArrayField(TEXT("ids_with_special")));
			C.IdsWithoutSpecial = ReadIntArray(O->GetArrayField(TEXT("ids_without_special")));
			Out.Cases.Add(MoveTemp(C));
		}
		const TArray<TSharedPtr<FJsonValue>>* BatchArr = nullptr;
		if (Root->TryGetArrayField(TEXT("batches"), BatchArr))
		{
			for (const TSharedPtr<FJsonValue>& V : *BatchArr)
			{
				const TSharedPtr<FJsonObject> O = V->AsObject();
				FGoldenBatch B;
				B.Id = O->GetStringField(TEXT("id"));
				for (const TSharedPtr<FJsonValue>& S : O->GetArrayField(TEXT("case_ids")))
				{
					B.CaseIds.Add(S->AsString());
				}
				B.bAddSpecialTokens = O->GetBoolField(TEXT("add_special_tokens"));
				double Max = 0;
				if (O->TryGetNumberField(TEXT("max_length"), Max)) // null -> field is not a number -> stays 0
				{
					B.MaxLength = (int32)Max;
				}
				B.PadId = (int32)O->GetNumberField(TEXT("pad_id"));
				for (const TSharedPtr<FJsonValue>& R : O->GetArrayField(TEXT("input_ids")))
				{
					B.InputIds.Add(ReadIntArray(R->AsArray()));
				}
				for (const TSharedPtr<FJsonValue>& R : O->GetArrayField(TEXT("attention_mask")))
				{
					B.AttentionMask.Add(ReadIntArray(R->AsArray()));
				}
				Out.Batches.Add(MoveTemp(B));
			}
		}
		return true;
	}

	static bool LoadAllGoldens(FAutomationTestBase& Test, TArray<FGolden>& Out)
	{
		bool bOk = true;
		for (const TCHAR* Name : GoldenNames)
		{
			FGolden G;
			if (LoadGolden(Test, Name, G))
			{
				Out.Add(MoveTemp(G));
			}
			else
			{
				bOk = false;
			}
		}
		return bOk;
	}

	static FString IdsToString(const TArray<int32>& Ids)
	{
		FString S;
		for (int32 i = 0; i < Ids.Num() && i < 24; ++i)
		{
			S += FString::Printf(TEXT("%d "), Ids[i]);
		}
		return S + (Ids.Num() > 24 ? TEXT("...") : TEXT(""));
	}

	static TStrongObjectPtr<UPipelineTokenizer> MakeTokenizer(const FString& Json, bool* bOutOk = nullptr)
	{
		TStrongObjectPtr<UPipelineTokenizer> Tok(NewObject<UPipelineTokenizer>(GetTransientPackage()));
		const bool bOk = Tok->InitializeTokenizerFromJson(Json);
		if (bOutOk)
		{
			*bOutOk = bOk;
		}
		return Tok;
	}

	/** ProcessEvent helpers for UTokenizerWrapper (reached through reflection, no build dependency). */
	static bool WrapperInit(UObject* Wrapper, const FString& Json)
	{
		UFunction* Func = Wrapper->FindFunction(TEXT("InitializeTokenizerFromJson"));
		if (!Func)
		{
			return false;
		}
		FStructOnScope Params(Func);
		FStrProperty* In = FindFProperty<FStrProperty>(Func, TEXT("JsonBlob"));
		FBoolProperty* Ret = FindFProperty<FBoolProperty>(Func, TEXT("ReturnValue"));
		if (!In || !Ret)
		{
			return false;
		}
		In->SetPropertyValue_InContainer(Params.GetStructMemory(), Json);
		Wrapper->ProcessEvent(Func, Params.GetStructMemory());
		return Ret->GetPropertyValue_InContainer(Params.GetStructMemory());
	}

	static bool WrapperEncode(UObject* Wrapper, const FString& Text, bool bSpecial, TArray<int32>& OutIds)
	{
		UFunction* Func = Wrapper->FindFunction(TEXT("Encode"));
		if (!Func)
		{
			return false;
		}
		FStructOnScope Params(Func);
		FStrProperty* In = FindFProperty<FStrProperty>(Func, TEXT("Text"));
		FBoolProperty* Flag = FindFProperty<FBoolProperty>(Func, TEXT("bAddSpecialTokens"));
		FArrayProperty* Ret = FindFProperty<FArrayProperty>(Func, TEXT("ReturnValue"));
		if (!In || !Flag || !Ret)
		{
			return false;
		}
		uint8* Mem = Params.GetStructMemory();
		In->SetPropertyValue_InContainer(Mem, Text);
		Flag->SetPropertyValue_InContainer(Mem, bSpecial);
		Wrapper->ProcessEvent(Func, Mem);
		FScriptArrayHelper Helper(Ret, Ret->ContainerPtrToValuePtr<void>(Mem));
		FIntProperty* Inner = CastField<FIntProperty>(Ret->Inner);
		if (!Inner)
		{
			return false;
		}
		OutIds.Reset();
		for (int32 i = 0; i < Helper.Num(); ++i)
		{
			OutIds.Add(Inner->GetPropertyValue(Helper.GetRawPtr(i)));
		}
		return true;
	}

	/** Runs NumThreads workers; each calls Encode on its tokenizer NumIters times and counts mismatches vs the golden. */
	static int32 RunThreads(const TArray<UPipelineTokenizer*>& TokPerThread, const FGolden& G, int32 NumIters, bool& bOutTimedOut)
	{
		TArray<TFuture<int32>> Futures;
		for (int32 T = 0; T < TokPerThread.Num(); ++T)
		{
			UPipelineTokenizer* Tok = TokPerThread[T];
			const FGolden* GP = &G;
			Futures.Add(Async(EAsyncExecution::ThreadPool, [Tok, GP, NumIters, T]()
			{
				int32 Mismatches = 0;
				for (int32 i = 0; i < NumIters; ++i)
				{
					const FGoldenCase& C = GP->Cases[(i + T) % GP->Cases.Num()];
					const bool bSpecial = (i & 1) != 0;
					const TArray<int32> Ids = Tok->Encode(C.Text, bSpecial);
					if (Ids != (bSpecial ? C.IdsWithSpecial : C.IdsWithoutSpecial))
					{
						++Mismatches;
					}
				}
				return Mismatches;
			}));
		}
		int32 Total = 0;
		bOutTimedOut = false;
		const FDateTime Deadline = FDateTime::UtcNow() + FTimespan::FromSeconds(60);
		for (TFuture<int32>& F : Futures)
		{
			const FTimespan Remaining = Deadline - FDateTime::UtcNow();
			if (Remaining <= FTimespan::Zero() || !F.WaitFor(Remaining))
			{
				bOutTimedOut = true;
				return Total;
			}
			Total += F.Get();
		}
		return Total;
	}
}

using namespace PipelineTokenizerSpecPrivate;

BEGIN_DEFINE_SPEC(FPipelineTokenizerSpec, "Pipelines.Tokenizer",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
END_DEFINE_SPEC(FPipelineTokenizerSpec)

void FPipelineTokenizerSpec::Define()
{
	Describe("Parity", [this]()
	{
		It("Encode matches the Python golden ids with and without special tokens (all cases, incl. non-ASCII and emoji)", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			for (const FGolden& G : Goldens)
			{
				TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(G.TokenizerJson);
				TestTrue(*FString::Printf(TEXT("%s: IsInitialized expected true"), *G.Name), Tok->IsInitialized());
				TestTrue(*FString::Printf(TEXT("%s: golden has cases"), *G.Name), G.Cases.Num() > 0);
				int32 Bad = 0;
				for (const FGoldenCase& C : G.Cases)
				{
					// A byte-level BPE or normalizer that mishandled UTF-8 (UTF-16 vs UTF-8 conversion, surrogate pairs
					// in emoji) would differ on the non-ASCII cases; specials flag mixups differ on every BERT/T5 case.
					const TArray<int32> With = Tok->Encode(C.Text, true);
					const TArray<int32> Without = Tok->Encode(C.Text, false);
					if (With != C.IdsWithSpecial)
					{
						++Bad;
						AddError(FString::Printf(TEXT("%s/%s with-special mismatch: expected [%s] got [%s]"),
							*G.Name, *C.Id, *IdsToString(C.IdsWithSpecial), *IdsToString(With)));
					}
					if (Without != C.IdsWithoutSpecial)
					{
						++Bad;
						AddError(FString::Printf(TEXT("%s/%s without-special mismatch: expected [%s] got [%s]"),
							*G.Name, *C.Id, *IdsToString(C.IdsWithoutSpecial), *IdsToString(Without)));
					}
				}
				TestEqual(*FString::Printf(TEXT("%s: mismatching encodes over %d cases (expected 0)"), *G.Name, G.Cases.Num()), Bad, 0);
			}
		});

		It("Encode of the empty string matches the golden (no crash, no stray ids)", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			for (const FGolden& G : Goldens)
			{
				const FGoldenCase* Empty = G.Cases.FindByPredicate([](const FGoldenCase& C) { return C.Text.IsEmpty(); });
				if (!Empty)
				{
					AddError(FString::Printf(TEXT("%s: golden has no empty-string case"), *G.Name));
					continue;
				}
				TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(G.TokenizerJson);
				TestTrue(*FString::Printf(TEXT("%s: empty text, special: ids equal golden"), *G.Name),
					Tok->Encode(FString(), true) == Empty->IdsWithSpecial);
				TestTrue(*FString::Printf(TEXT("%s: empty text, plain: ids equal golden"), *G.Name),
					Tok->Encode(FString(), false) == Empty->IdsWithoutSpecial);
			}
		});

		It("repeated Encode of the same text is deterministic (no state carried between calls)", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			for (const FGolden& G : Goldens)
			{
				TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(G.TokenizerJson);
				// Walk the cases backwards then forwards: a tokenizer that reused a stale result buffer would leak
				// the previous row into the next.
				int32 Bad = 0;
				for (int32 Pass = 0; Pass < 2; ++Pass)
				{
					for (int32 k = 0; k < G.Cases.Num(); ++k)
					{
						const FGoldenCase& C = G.Cases[Pass == 0 ? G.Cases.Num() - 1 - k : k];
						Bad += (Tok->Encode(C.Text, true) != C.IdsWithSpecial) ? 1 : 0;
					}
				}
				TestEqual(*FString::Printf(TEXT("%s: mismatches over two passes (expected 0)"), *G.Name), Bad, 0);
			}
		});
	});

	Describe("VocabSize", [this]()
	{
		It("GetVocabSize equals the golden vocab_size_with_added and is stable", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			for (const FGolden& G : Goldens)
			{
				TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(G.TokenizerJson);
				const int32 V1 = Tok->GetVocabSize();
				const int32 V2 = Tok->GetVocabSize();
				if (G.VocabSize > 0)
				{
					TestEqual(*FString::Printf(TEXT("%s: GetVocabSize expected %d (golden)"), *G.Name, G.VocabSize), V1, G.VocabSize);
				}
				else
				{
					TestTrue(*FString::Printf(TEXT("%s: GetVocabSize expected > 0, got %d"), *G.Name, V1), V1 > 0);
				}
				TestEqual(*FString::Printf(TEXT("%s: GetVocabSize stable across calls"), *G.Name), V2, V1);
			}
		});
	});

	Describe("Batch", [this]()
	{
		It("EncodeBatch reproduces every golden batch (rows, right padding, mask, MaxLength truncation)", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			for (const FGolden& G : Goldens)
			{
				TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(G.TokenizerJson);
				TestTrue(*FString::Printf(TEXT("%s: golden has batches"), *G.Name), G.Batches.Num() > 0);
				for (const FGoldenBatch& B : G.Batches)
				{
					const FString Tag = FString::Printf(TEXT("%s/%s"), *G.Name, *B.Id);
					TArray<FString> Texts;
					for (const FString& Id : B.CaseIds)
					{
						const FGoldenCase* C = G.FindCase(Id);
						if (!C) { AddError(FString::Printf(TEXT("%s: golden batch references unknown case %s"), *Tag, *Id)); continue; }
						Texts.Add(C->Text);
					}
					FPipelineTokenizedBatch Out;
					// Pre-fill with garbage: a batch not reset between calls would keep it.
					Out.InputIds = {7, 7, 7};
					Out.AttentionMask = {7};
					Out.NumRows = 99;
					Out.SeqLen = 99;
					const bool bOk = Tok->EncodeBatch(Texts, Out, B.bAddSpecialTokens, B.MaxLength, B.PadId);
					TestTrue(*FString::Printf(TEXT("%s: EncodeBatch expected true (error: %s)"), *Tag, *Tok->GetLastError()), bOk);
					const int32 ExpRows = B.InputIds.Num();
					const int32 ExpLen = ExpRows > 0 ? B.InputIds[0].Num() : 0;
					TestEqual(*FString::Printf(TEXT("%s: NumRows"), *Tag), Out.NumRows, ExpRows);
					// SeqLen is the longest row after truncation: a SeqLen = MaxLength implementation fails when all rows are shorter.
					TestEqual(*FString::Printf(TEXT("%s: SeqLen (max_length=%d)"), *Tag, B.MaxLength), Out.SeqLen, ExpLen);
					TestEqual(*FString::Printf(TEXT("%s: InputIds length"), *Tag), Out.InputIds.Num(), ExpRows * ExpLen);
					TestEqual(*FString::Printf(TEXT("%s: AttentionMask length"), *Tag), Out.AttentionMask.Num(), ExpRows * ExpLen);
					if (Out.InputIds.Num() != ExpRows * ExpLen || Out.AttentionMask.Num() != ExpRows * ExpLen) { continue; }
					int32 BadIds = 0, BadMask = 0;
					for (int32 r = 0; r < ExpRows; ++r)
					{
						for (int32 c = 0; c < ExpLen; ++c)
						{
							BadIds += Out.InputIds[r * ExpLen + c] != B.InputIds[r][c] ? 1 : 0;
							// Mask must be 0 exactly on padding even when a real token id equals PadTokenId (gpt2 <|endoftext|>).
							BadMask += Out.AttentionMask[r * ExpLen + c] != B.AttentionMask[r][c] ? 1 : 0;
						}
					}
					TestEqual(*FString::Printf(TEXT("%s: input id mismatches (expected 0)"), *Tag), BadIds, 0);
					TestEqual(*FString::Printf(TEXT("%s: attention mask mismatches (expected 0)"), *Tag), BadMask, 0);
				}
			}
		});

		It("EncodeBatch rows equal per-text Encode, padded with a custom PadTokenId to the longest row", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			for (const FGolden& G : Goldens)
			{
				TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(G.TokenizerJson);
				// Short, medium and long golden cases in a different order than the golden batches use.
				TArray<FString> Texts;
				TArray<const FGoldenCase*> Cs;
				for (const TCHAR* Id : { TEXT("s01"), TEXT("s00"), TEXT("s20") })
				{
					const FGoldenCase* C = G.FindCase(Id);
					if (!C) { AddError(FString::Printf(TEXT("%s: golden case %s missing"), *G.Name, Id)); return; }
					Cs.Add(C);
					Texts.Add(C->Text);
				}
				const int32 Pad = 31337; // distinct from every real id and from the default 0
				FPipelineTokenizedBatch Out;
				TestTrue(*FString::Printf(TEXT("%s: EncodeBatch true"), *G.Name), Tok->EncodeBatch(Texts, Out, true, 0, Pad));
				int32 Longest = 0;
				for (const FGoldenCase* C : Cs) { Longest = FMath::Max(Longest, C->IdsWithSpecial.Num()); }
				TestEqual(*FString::Printf(TEXT("%s: SeqLen expected longest row %d"), *G.Name, Longest), Out.SeqLen, Longest);
				TestEqual(*FString::Printf(TEXT("%s: NumRows expected 3"), *G.Name), Out.NumRows, 3);
				if (Out.InputIds.Num() != 3 * Longest || Out.AttentionMask.Num() != 3 * Longest)
				{
					AddError(FString::Printf(TEXT("%s: wrong buffer sizes %d / %d, expected %d"), *G.Name, Out.InputIds.Num(), Out.AttentionMask.Num(), 3 * Longest));
					continue;
				}
				for (int32 r = 0; r < 3; ++r)
				{
					const TArray<int32>& Exp = Cs[r]->IdsWithSpecial;
					for (int32 c = 0; c < Longest; ++c)
					{
						const bool bReal = c < Exp.Num();
						TestEqual(*FString::Printf(TEXT("%s row %d col %d id"), *G.Name, r, c), Out.InputIds[r * Longest + c], bReal ? Exp[c] : Pad);
						TestEqual(*FString::Printf(TEXT("%s row %d col %d mask"), *G.Name, r, c), Out.AttentionMask[r * Longest + c], bReal ? 1 : 0);
					}
				}
			}
		});

		It("EncodeBatch MaxLength truncation counts special tokens and cuts every row to at most MaxLength", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			for (const FGolden& G : Goldens)
			{
				TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(G.TokenizerJson);
				const FGoldenCase* Long = G.FindCase(TEXT("s01")); // "The quick brown fox ..." > 4 tokens everywhere
				if (!Long) { AddError(TEXT("golden case s01 missing")); return; }
				const int32 Max = 4;
				TestTrue(*FString::Printf(TEXT("%s: s01 has more than %d ids"), *G.Name, Max), Long->IdsWithSpecial.Num() > Max);
				FPipelineTokenizedBatch Out;
				TestTrue(TEXT("EncodeBatch with MaxLength 4 expected true"), Tok->EncodeBatch({ Long->Text }, Out, true, Max, 0));
				TestEqual(*FString::Printf(TEXT("%s: SeqLen expected %d"), *G.Name, Max), Out.SeqLen, Max);
				TestEqual(*FString::Printf(TEXT("%s: NumRows expected 1"), *G.Name), Out.NumRows, 1);
				// Truncation keeps the first Max ids of the special-token-inclusive encoding for single-sequence
				// truncation of the right side, except where the tokenizer's own truncation re-adds the closing
				// special token. The goldens' b01/b02 cover exact ids; here we only pin the length and mask.
				if (Out.AttentionMask.Num() == Max)
				{
					for (int32 c = 0; c < Max; ++c)
					{
						TestEqual(*FString::Printf(TEXT("%s: mask[%d] expected 1 (no padding in a truncated row)"), *G.Name, c), Out.AttentionMask[c], 1);
					}
				}
				else
				{
					AddError(FString::Printf(TEXT("%s: AttentionMask length %d expected %d"), *G.Name, Out.AttentionMask.Num(), Max));
				}
			}
		});

		It("EncodeBatch with empty Texts returns true and a 0x0 batch (previous contents cleared)", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(Goldens[0].TokenizerJson);
			FPipelineTokenizedBatch Out;
			Out.InputIds = {1, 2, 3};
			Out.AttentionMask = {1, 1, 1};
			Out.NumRows = 1;
			Out.SeqLen = 3;
			TestTrue(TEXT("EncodeBatch({}) expected true"), Tok->EncodeBatch(TArray<FString>(), Out, true, 0, 0));
			TestEqual(TEXT("NumRows expected 0"), Out.NumRows, 0);
			TestEqual(TEXT("SeqLen expected 0"), Out.SeqLen, 0);
			TestEqual(TEXT("InputIds expected empty"), Out.InputIds.Num(), 0);
			TestEqual(TEXT("AttentionMask expected empty"), Out.AttentionMask.Num(), 0);
		});

		It("EncodeBatch rejects MaxLength -1 and PadTokenId -1: false, batch reset, GetLastError set", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(Goldens[0].TokenizerJson);
			// One warning per failed call is the C42 contract; the prefix "UPipelineTokenizer::" is an assumption
			// about the module's message text (the same convention UTokenizerWrapper uses).
			AddExpectedMessagePlain(TEXT("UPipelineTokenizer::"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 2);
			for (int32 Case = 0; Case < 2; ++Case)
			{
				const int32 MaxLength = Case == 0 ? -1 : 0;
				const int32 PadId = Case == 1 ? -1 : 0;
				FPipelineTokenizedBatch Out;
				Out.InputIds = {5, 5};
				Out.AttentionMask = {1, 1};
				Out.NumRows = 2;
				Out.SeqLen = 1;
				const FString What = Case == 0 ? TEXT("MaxLength -1") : TEXT("PadTokenId -1");
				TestFalse(*FString::Printf(TEXT("%s: EncodeBatch expected false"), *What), Tok->EncodeBatch({ TEXT("Hello") }, Out, false, MaxLength, PadId));
				// A stale batch after failure would be fed to inference by a careless caller.
				TestEqual(*FString::Printf(TEXT("%s: NumRows reset to 0"), *What), Out.NumRows, 0);
				TestEqual(*FString::Printf(TEXT("%s: SeqLen reset to 0"), *What), Out.SeqLen, 0);
				TestEqual(*FString::Printf(TEXT("%s: InputIds reset"), *What), Out.InputIds.Num(), 0);
				TestEqual(*FString::Printf(TEXT("%s: AttentionMask reset"), *What), Out.AttentionMask.Num(), 0);
				TestFalse(*FString::Printf(TEXT("%s: GetLastError expected non-empty"), *What), Tok->GetLastError().IsEmpty());
			}
			// The object remains usable and the error clears after a good call.
			FPipelineTokenizedBatch Good;
			TestTrue(TEXT("EncodeBatch valid after failures"), Tok->EncodeBatch({ TEXT("Hello, world!") }, Good, false, 0, 0));
			TestTrue(TEXT("GetLastError expected empty after a successful call"), Tok->GetLastError().IsEmpty());
		});
	});

	Describe("Errors", [this]()
	{
		It("an uninitialized tokenizer returns empty/-1/false and sets GetLastError", [this]()
		{
			TStrongObjectPtr<UPipelineTokenizer> Tok(NewObject<UPipelineTokenizer>(GetTransientPackage()));
			TestFalse(TEXT("IsInitialized expected false"), Tok->IsInitialized());
			// Three failing calls -> three warnings (assumed text prefix "UPipelineTokenizer::").
			AddExpectedMessagePlain(TEXT("UPipelineTokenizer::"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 3);
			TestEqual(TEXT("Encode on uninitialized expected empty"), Tok->Encode(TEXT("Hello"), true).Num(), 0);
			TestFalse(TEXT("GetLastError after Encode expected non-empty"), Tok->GetLastError().IsEmpty());
			TestEqual(TEXT("GetVocabSize on uninitialized expected -1"), Tok->GetVocabSize(), -1);
			FPipelineTokenizedBatch Out;
			Out.NumRows = 4;
			TestFalse(TEXT("EncodeBatch on uninitialized expected false"), Tok->EncodeBatch({ TEXT("a") }, Out));
			TestEqual(TEXT("batch reset to 0 rows after failure"), Out.NumRows, 0);
			TestFalse(TEXT("GetLastError after EncodeBatch expected non-empty"), Tok->GetLastError().IsEmpty());
		});

		It("invalid JSON returns false and the previously loaded tokenizer keeps working", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			const FGolden& G = Goldens[0];
			TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(G.TokenizerJson);
			const FGoldenCase* C = G.FindCase(TEXT("s00"));
			if (!C) { AddError(TEXT("golden case s00 missing")); return; }
			AddExpectedMessagePlain(TEXT("UPipelineTokenizer::"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 2);
			TestFalse(TEXT("garbage text expected false"), Tok->InitializeTokenizerFromJson(TEXT("this is not json {")));
			TestFalse(TEXT("GetLastError expected non-empty after failed init"), Tok->GetLastError().IsEmpty());
			TestTrue(TEXT("IsInitialized must stay true (keep-previous-on-failure)"), Tok->IsInitialized());
			TestTrue(TEXT("Encode still equals golden after failed re-init"), Tok->Encode(C->Text, true) == C->IdsWithSpecial);
			TestFalse(TEXT("empty JSON expected false"), Tok->InitializeTokenizerFromJson(FString()));
			TestTrue(TEXT("Encode still equals golden after second failed re-init"), Tok->Encode(C->Text, true) == C->IdsWithSpecial);
			TestTrue(TEXT("GetLastError expected empty after a successful Encode"), Tok->GetLastError().IsEmpty());
		});

		It("a failed first init leaves the object uninitialized", [this]()
		{
			TStrongObjectPtr<UPipelineTokenizer> Tok(NewObject<UPipelineTokenizer>(GetTransientPackage()));
			AddExpectedMessagePlain(TEXT("UPipelineTokenizer::"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
			TestFalse(TEXT("invalid JSON expected false"), Tok->InitializeTokenizerFromJson(TEXT("{\"model\": 1}")));
			TestFalse(TEXT("IsInitialized expected false"), Tok->IsInitialized());
			TestFalse(TEXT("GetLastError expected non-empty"), Tok->GetLastError().IsEmpty());
		});

		It("accepts a leading U+FEFF byte order mark and GetLastError is empty after success", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			for (const FGolden& G : Goldens)
			{
				FString WithBom;
				WithBom.AppendChar((TCHAR)0xFEFF);
				WithBom += G.TokenizerJson;
				bool bOk = false;
				TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(WithBom, &bOk);
				TestTrue(*FString::Printf(TEXT("%s: BOM-prefixed JSON expected to load (error: %s)"), *G.Name, *Tok->GetLastError()), bOk);
				TestTrue(*FString::Printf(TEXT("%s: GetLastError expected empty after successful init"), *G.Name), Tok->GetLastError().IsEmpty());
				const FGoldenCase* C = G.FindCase(TEXT("s01"));
				if (C && bOk)
				{
					TestTrue(*FString::Printf(TEXT("%s: Encode after BOM load equals golden"), *G.Name), Tok->Encode(C->Text, true) == C->IdsWithSpecial);
				}
			}
		});

		It("re-initializing with another valid tokenizer switches to it", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			TStrongObjectPtr<UPipelineTokenizer> Tok(NewObject<UPipelineTokenizer>(GetTransientPackage()));
			for (int32 Round = 0; Round < Goldens.Num() * 2; ++Round)
			{
				const FGolden& G = Goldens[Round % Goldens.Num()];
				TestTrue(*FString::Printf(TEXT("init %s expected true"), *G.Name), Tok->InitializeTokenizerFromJson(G.TokenizerJson));
				const FGoldenCase* C = G.FindCase(TEXT("s02"));
				if (C)
				{
					TestTrue(*FString::Printf(TEXT("%s: Encode after re-init equals golden (round %d)"), *G.Name, Round), Tok->Encode(C->Text, true) == C->IdsWithSpecial);
				}
			}
		});
	});

	Describe("Threads", [this]()
	{
		It("8 threads x 200 Encode calls on ONE tokenizer all equal the golden ids", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			for (const FGolden& G : Goldens)
			{
				TStrongObjectPtr<UPipelineTokenizer> Tok = MakeTokenizer(G.TokenizerJson);
				TArray<UPipelineTokenizer*> PerThread;
				for (int32 i = 0; i < 8; ++i) { PerThread.Add(Tok.Get()); }
				bool bTimedOut = false;
				const int32 Mismatches = RunThreads(PerThread, G, 200, bTimedOut);
				TestFalse(*FString::Printf(TEXT("%s: workers finished within 60 s (a deadlock would time out)"), *G.Name), bTimedOut);
				TestEqual(*FString::Printf(TEXT("%s: mismatching results across 1600 calls on a shared tokenizer (expected 0)"), *G.Name), Mismatches, 0);
			}
		});

		It("8 threads x 200 Encode calls on 8 SEPARATE tokenizers all equal the golden ids", [this]()
		{
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			for (const FGolden& G : Goldens)
			{
				TArray<TStrongObjectPtr<UPipelineTokenizer>> Owners;
				TArray<UPipelineTokenizer*> PerThread;
				for (int32 i = 0; i < 8; ++i)
				{
					Owners.Add(MakeTokenizer(G.TokenizerJson));
					PerThread.Add(Owners.Last().Get());
				}
				bool bTimedOut = false;
				const int32 Mismatches = RunThreads(PerThread, G, 200, bTimedOut);
				TestFalse(*FString::Printf(TEXT("%s: workers finished within 60 s"), *G.Name), bTimedOut);
				TestEqual(*FString::Printf(TEXT("%s: mismatching results across 1600 calls on separate tokenizers (expected 0)"), *G.Name), Mismatches, 0);
			}
		});
	});

	Describe("CoexistsWithPlugin", [this]()
	{
		It("UTokenizerWrapper (Tokenizers plugin) and UPipelineTokenizer give identical ids and are independent", [this]()
		{
			if (!FModuleManager::Get().IsModuleLoaded("Tokenizers"))
			{
				AddError(TEXT("The 'Tokenizers' module is not loaded. PipelinesTestProject must enable the Tokenizers plugin so both lib copies are exercised in one process."));
				return;
			}
			UClass* WrapperClass = FindObject<UClass>(nullptr, TEXT("/Script/Tokenizers.TokenizerWrapper"));
			if (!WrapperClass)
			{
				AddError(TEXT("Class /Script/Tokenizers.TokenizerWrapper not found although the Tokenizers module is loaded."));
				return;
			}
			TArray<FGolden> Goldens;
			if (!LoadAllGoldens(*this, Goldens)) { return; }
			for (const FGolden& G : Goldens)
			{
				TStrongObjectPtr<UObject> Wrapper(NewObject<UObject>(GetTransientPackage(), WrapperClass));
				if (!WrapperInit(Wrapper.Get(), G.TokenizerJson))
				{
					AddError(FString::Printf(TEXT("%s: UTokenizerWrapper::InitializeTokenizerFromJson failed or was not reachable by reflection"), *G.Name));
					continue;
				}
				TStrongObjectPtr<UPipelineTokenizer> Mine = MakeTokenizer(G.TokenizerJson);
				int32 Bad = 0;
				for (const FGoldenCase& C : G.Cases)
				{
					TArray<int32> FromPlugin;
					if (!WrapperEncode(Wrapper.Get(), C.Text, true, FromPlugin)) { AddError(TEXT("Encode reflection call failed")); return; }
					const TArray<int32> FromMine = Mine->Encode(C.Text, true);
					if (FromPlugin != FromMine || FromMine != C.IdsWithSpecial)
					{
						++Bad;
						AddError(FString::Printf(TEXT("%s/%s: plugin [%s] vs module [%s] vs golden [%s]"), *G.Name, *C.Id,
							*IdsToString(FromPlugin), *IdsToString(FromMine), *IdsToString(C.IdsWithSpecial)));
					}
				}
				TestEqual(*FString::Printf(TEXT("%s: cases where the two copies disagree (expected 0)"), *G.Name), Bad, 0);

				const FGoldenCase* Probe = G.FindCase(TEXT("s01"));
				if (!Probe) { continue; }

				// Destroy the module's tokenizer; the plugin's copy (separate lib symbols/heap) must keep working.
				Mine.Reset();
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
				TArray<int32> After;
				TestTrue(*FString::Printf(TEXT("%s: plugin Encode callable after module tokenizer destroyed"), *G.Name), WrapperEncode(Wrapper.Get(), Probe->Text, true, After));
				TestTrue(*FString::Printf(TEXT("%s: plugin ids unchanged after module tokenizer destroyed"), *G.Name), After == Probe->IdsWithSpecial);

				// And the other way round.
				TStrongObjectPtr<UPipelineTokenizer> Mine2 = MakeTokenizer(G.TokenizerJson);
				Wrapper.Reset();
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
				TestTrue(*FString::Printf(TEXT("%s: module Encode unchanged after plugin tokenizer destroyed"), *G.Name),
					Mine2->Encode(Probe->Text, true) == Probe->IdsWithSpecial);
			}
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
