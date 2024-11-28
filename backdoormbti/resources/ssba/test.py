import subprocess

if __name__ == "__main__":

    data = "cifar10"
    celeba = 1
    subprocess.call(["bash", "poison_data.sh", data, str(celeba)])
