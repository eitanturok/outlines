from outlines import models, generate
from icecream import install
install()

def main():

    model = models.tinygradlm("gpt2")
    generator = generate.text(model)
    answer = generator("My favorite puppy is")
    ic(answer)

if __name__ == '__main__':
    main()
