import os




try:
    os.system('python3 src/preprocessing.py')
except Exception as e:
    print(e)
    print('Failed to preprocess the dataset.')
    exit(1)
