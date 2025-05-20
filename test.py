from outlines import models, generate
from icecream import install
install()

def main():

    model = models.tinygradlm("Qwen/QwQ-32B-Preview")
    generator = generate.text(model)
    answer = generator("A prompt", temperature=2.0)
    ic(answer)

if __name__ == '__main__':
    main()
