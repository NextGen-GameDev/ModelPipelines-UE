# Install

You need:

- Windows 64-bit.
- Unreal Engine 5.8 or 5.7. (5.6 is planned, not tested.)
- Visual Studio 2022 with the "Game development with C++" workload, if you install from source (see Option A).

Model Pipelines needs two other plugins:

- **Tokenizers**: a separate plugin that turns text into numbers for the model. You must install it too.
- **NNERuntimeORT**: an engine plugin. Unreal enables it for you.

## Option A: install from source into a C++ project

1. Close the Unreal Editor.
2. Open your project folder (the one with the `.uproject` file).
3. Create a folder named `Plugins` if it does not exist.
4. Copy the `ModelPipelines` plugin folder into it, so `ModelPipelines.uplugin` is at `<Project>/Plugins/ModelPipelines/ModelPipelines.uplugin`.
5. Copy the `Tokenizers` plugin folder into `Plugins` too, so `Tokenizers.uplugin` is at `<Project>/Plugins/Tokenizers/Tokenizers.uplugin`. (A folder with another name, such as `Tokenizers-UE`, also works.)
6. Check that this file exists: `Plugins/Tokenizers/Source/ThirdParty/tokenizersLibrary/x64/Release/tokenizers_c.lib`.
   - Release downloads include it. If you copied from a source checkout and it is missing, go to "Build tokenizers_c.lib" below.
7. Right-click the `.uproject` file and choose **Generate Visual Studio project files**.
8. Double-click the `.uproject` file. The editor asks to rebuild the missing modules. Click **Yes**.
   - You should see a build progress window. When it ends, the editor opens.
9. In the editor, open **Edit > Plugins**.
10. Search for `Model Pipelines`. You find **Model Pipelines** under **AI**. Make sure its box is ticked.
11. Search for `Tokenizers`. Make sure **Tokenizers** is ticked too.
12. If you ticked anything, click **Restart Now**.

You should now see **Model Pipelines** in the Plugins window with a tick, and a new plugin content folder in the Content Browser (see below).

### Build tokenizers_c.lib

Do this only if step 6 failed. You need Rust and CMake installed.

1. Open PowerShell 5.1 in the `Plugins/Tokenizers` folder.
2. Run:

   ```powershell
   .\Scripts\BuildTokenizersLib.ps1
   ```

3. Wait about 5 minutes. When it finishes, `tokenizers_c.lib` is in place.

## Option B: Blueprint-only project

A Blueprint-only project cannot compile plugin source. Pick one:

- Use a **prebuilt** copy of both plugins (a plugin folder that already has a `Binaries` folder for your engine version). Copy both folders into `<Project>/Plugins/` and open the project.
- Or turn the project into a C++ project: in the editor choose **Tools > New C++ Class...**, pick **None**, click **Next**, then **Create Class**. Wait for Visual Studio and the build. Then follow Option A from step 1.

## Show the sample content

The sample maps live in the plugin's own Content folder. The Content Browser hides plugin content by default.

1. Open the Content Browser.
2. Click **Settings** (top right of the Content Browser).
3. Tick **Show Plugin Content**.

You should now see a **Plugins** section in the folder tree. Open the ModelPipelines plugin's content folder, then **Samples**, then **Maps**, to find the four sample maps.

## Check that it works

1. Open **Edit > Plugins** and search for `Model Pipelines`. It is ticked.
2. Right-click a folder in the Content Browser. You should see a **Model Pipelines** section in the menu (see [Import a model](ImportModel.md)).

Next: [Import a model](ImportModel.md).
