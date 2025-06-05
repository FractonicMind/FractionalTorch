📚 FractionalTorch

FractionalTorch is an experimental PyTorch module that implements NeuroFraction-of-1.5, a creative approach to generative AI that embraces glitch, artifacts, and creative noise.

By introducing fractional dropout—intentional, controlled chaos—FractionalTorch aims to transform mistakes into features, enabling more dynamic, human-like real-time video generation.
✨ Why NeuroFraction-of-1.5?

Perfect AI is boring. NeuroFraction-of-1.5 introduces fractional dropout to create controlled chaos in real-time video generation—transforming mistakes into features. Think glitch art, jazz improvisation, and creative noise that breathes life into generative models.
🚀 Quick Example

import torch
from fractionaldropout import FractionalDropout

model = MyModel()
model.add_module('fractional_dropout', FractionalDropout(p=0.33))

🛠️ Installation

Clone the repository:

git clone https://github.com/FractonicMind/FractionalTorch.git
cd FractionalTorch

Install requirements (currently just PyTorch):

pip install torch

🎥 Usage Example

import torch
from fractionaldropout import FractionalDropout

dropout = FractionalDropout(p=0.33)
x = torch.randn(2, 3, 4, 4)  # Example tensor
output = dropout(x)
print(output.shape)

📈 Roadmap / TODO

    Integrate with real-time video pipeline

    Explore temporal coherence techniques

    Add unit tests and documentation

    Create demo notebooks and sample videos

🤝 Contributing

Contributions are welcome! Please open an issue or pull request to discuss your ideas or improvements. Let’s glitch the future together.
📄 License

MIT License. See LICENSE for details.
🔗 Links

    [Medium Article](https://medium.com/@leogouk/where-ai-crafts-video-in-real-time-the-neurofraction-of-1-5-revolution-b0d2d6b2aa30)

