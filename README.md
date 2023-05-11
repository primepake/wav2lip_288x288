# This is a 288x288 wav2lip model version.
The original repo: https://github.com/Rudrabha/Wav2Lip
Some Features I will implement here
- [x] input size 288x288
- [x] PRelu
- [x] LeakyRelu
- [x] Gradient penalty
- [x] Wasserstein Loss
- [] wav2lip_384
- [] wav2lip_512
- [] syncnet_192
- [] syncnet_384
- [] 2TUnet instead of simple unet in wav2lip original: https://arxiv.org/abs/2210.15374
- [] MSG-UNet: https://github.com/laxmaniron/MSG-U-Net
- [] SAM-UNet: https://github.com/1343744768/Multiattention-UNet
<br />
I trained my own model on AVSPEECH dataset and then transfer learning with my private dataset. 

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

