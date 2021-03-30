import pandas as pd
from k_means import KMeans
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("iris.csv")
    print(df)

    clean_df = df.drop(['flower', 'petal_width'], axis=1)
    print(clean_df)

    kmeans = KMeans(3, clean_df)

    centroids, predictions = kmeans.fit()

    fig, ax = plt.subplots()

    for i, c in enumerate(['blue', 'orange', 'green']):
        sepal_length = predictions[predictions['cluster']==i]['sepal_length']
        sepal_width = predictions[predictions['cluster']==i]['sepal_width']
        ax.scatter(sepal_length, sepal_width, c=c)

    ax.set_xlabel('sepal_length')
    ax.set_ylabel('sepal_width')
    
    plt.savefig("segmentation.png")
    plt.show()



if __name__ == '__main__':
    main()