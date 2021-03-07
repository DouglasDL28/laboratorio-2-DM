import pandas as pd
from k_means import KMeans


def main():
    df = pd.read_csv("iris.csv")

    del df['flower']
    del df['petal_length']

    print(df)

    kmeans = KMeans(3, df)

    kmeans.fit()


if __name__ == '__main__':
    main()