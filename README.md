# Private Audio Transcriber with Translation Experiments
My experiments with adding a translation feature to the Private Audio Transcriber Console. The goal is to build intuition and gain practical experience.

- Testing different translation models
- Trying different architectures for local deployment - MLX, Ollama and hybrid approaches
  
Original Private Audio Transcriber Console project:<br>
https://github.com/vbookshelf/Private-Audio-Transcriber-Console


<br>

## Experiments
- <strong>Exp_1 - Using Whisper MLX and TranslateGemma 12b 4bit via Ollama</strong><br>
This is a working app. Uses a hybrid architecture: MLX plus Ollama. Whisper MLX does the transcription. The translations are done by TranslateGemma 12b 4bit. TranslateGemma is served using Ollama. The user needs to have Ollama installed for this app to work. Also the user will need to manually download the TranslateGemma model via Ollama.

- <strong>Exp_2 - Using Whisper MLX and Tiny Aya Global MLX</strong><br>
This is a working app. Uses MLX for both transcription and translation. Whisper MLX does the transcription. The translations are done by tiny-aya-global-8bit-mlx. Works well. Tiny Aya supports more than 60 languages. For the user, the installation is much simpler because there's no need to first install Ollama, and then also download Ollama models.  Both mlx models are auto downloaded from HuggingFace during installation.
