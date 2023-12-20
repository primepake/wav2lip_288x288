# This is a 288x288 wav2lip model version.
The original repo: https://github.com/Rudrabha/Wav2Lip
Some Features I will implement here
- [x] input size 288x288
- [x] PRelu
- [x] LeakyRelu
- [x] Gradient penalty
- [x] Wasserstein Loss
- [x] SAM-UNet: https://github.com/1343744768/Multiattention-UNet
<br />
Each line on filelist should be full path <br />
First, Train syncnet <br />

```
python3 train_syncnet_sam.py
```

Second, train wav2lip-Sam
```
python3 hq_wav2lip_sam_train.py
```
Some demo from chinese users:
https://github.com/primepake/wav2lip_288x288/issues/89#issue-2047907323
## Citing

To cite this repository:

```bibtex
@misc{Wav2Lip,
  author={Rudrabha},
  title={Wav2Lip: Accurately Lip-syncing Videos In The Wild},
  year={2020},
  url={https://github.com/Rudrabha/Wav2Lip}
}
```

