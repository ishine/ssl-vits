# SSL-VITS: A two-stage TTS model based on VITS, using SSL features as intermediate features.

画饼ing

In this case, I used [contentvec](https://github.com/auspicious3000/contentvec) as the SSL feature.In the future experiments, other SSL features will also be tested.



two stage:
+ text -> ssl units
+ ssl units -> wav

text -> ssl units modeling:
+ using CVAE architecture based on VITS with MAS and flow, where the HifiGAN decoder replaced by a SSL decoder. Adversarial training techniques have also been removed.


ssl units -> wav:
+ using [rvc](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI) based on vits

## References
+ [vits](https://github.com/jaywalnut310/vits)
+ [contentvec](https://github.com/auspicious3000/contentvec)
+ [multilanguage cleaner](https://github.com/CjangCjengh/vits)
+ [rvc](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
