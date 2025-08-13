# Better wav2lip model version.
Original repo: https://github.com/Rudrabha/Wav2Lip
- [x] model size 288x288, 384x384, 512x512
- [x] PRelu
- [x] LeakyRelu
- [x] Gradient penalty
- [x] Wasserstein Loss
- [x] SAM-UNet: https://github.com/1343744768/Multiattention-UNet
      
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

## Demo - 384x384 Resolution

### Quick Preview (GIF)
![Demo Preview](assets/test5_demo.gif)

### Full Quality Video
- [Download MOV (Original)](assets/test5_facial_dubbing_add_audio.mov)

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

