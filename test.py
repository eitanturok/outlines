from outlines import models, generate

def main():

    model = models.tinygradlm("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit")
    generator = generate.text(model)
    answer = generator("A prompt", temperature=2.0)

if __name__ == '__main__':
    main()
