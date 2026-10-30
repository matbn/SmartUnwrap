# SAM2 runtime

This folder contains the SAM2.1 Base+ checkpoint and only the official Python
modules/config required for image automatic-mask inference. The upstream
license terms are preserved in `LICENSE.txt`.
It omits notebooks, demos, training code, video predictors, other model
configurations, and the optional C++ extension.

The small `sam2/` package is vendored from Meta's official SAM2 repository at
commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
To enable inference, install the runtime dependencies in the active Python
environment. For CPU, or after selecting a PyTorch build matching the GPU,
install:

```powershell
pip install -r backend/third_party/SAM2/requirements.txt
```

The application automatically loads `sam2.1_hiera_base_plus.pt` from this
folder. The upstream source is at https://github.com/facebookresearch/sam2.
If the runtime dependencies are absent, the UI option remains available and reports
`sam2_mode: geometry fallback` in the API response.
