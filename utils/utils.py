
import os



def create_path(path):

    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == "__main__":
    create_path("/home/user/thesis_results/history")