# Contributing

Thanks for your interest in ModelPipelines.

## Contributor agreement

ModelPipelines is developed by NextGen-GameDev, which also builds paid products on top of
it. Before we can merge a contribution, you must agree to the
[Contributor License Agreement](CLA.md). To agree, add this line to the description of your
first pull request:

> I have read and agree to the ModelPipelines Contributor License Agreement (CLA.md), version 1.0.

and sign off every commit (`git commit -s`), which adds a `Signed-off-by:` line with your
name and email.

## Ground rules

- Win64 only; the code must build on every supported engine version (UE 5.7 and 5.8).
- Follow the Unreal Engine coding standard. No `std::` types in public headers, and no
  exceptions or RTTI in plugin modules.
- Never crash on user input: report errors through `FPipelineResult`.
- Don't commit model files, test data or golden outputs. Models must have a permissive
  license (Apache-2.0, MIT or BSD).
- Run the automation tests (`Automation RunTests Pipelines`) before opening a pull request.
