#!/usr/bin/python
import os

def main():
    os.system("diffusion_maps.py")
    os.sytem("nn.py embedding_train_x.pkl train_y.pkl")

if __name__ == '__main__':
    main()