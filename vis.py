import numpy as np
from matplotlib import pyplot as plt
# import matplotlib.cm as cm
import seaborn as sns
sns.set()
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

def loss_plot(loss):
    """ 学習記録(epoch-loss)をプロット """

    end_epoch = len(loss)

    plt.figure()
    runs = [i for i in range(1, end_epoch+1)]
    plt.plot(runs, loss, label='loss')

    plt.xlabel('step',fontsize=16)
    plt.ylabel('loss',fontsize=16)
    plt.xticks([i for i in range(1, end_epoch+1, 5)],fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=15)

    plt.savefig("loss.png", bbox_inches='tight')
    plt.close()

def acc_plot(acc):
    """ 学習記録(epoch-acc)をプロット """

    end_epoch = len(acc)

    plt.figure()
    runs = [i for i in range(1, end_epoch+1)]
    plt.plot(runs, acc, label='acc')

    plt.xlabel('step',fontsize=16)
    plt.ylabel('acc',fontsize=16)
    plt.xticks([i for i in range(1, end_epoch+1, 5)],fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=15)

    plt.savefig("acc.png", bbox_inches='tight')
    plt.close()

def scatter_plot(X, y, path, name):
    """ 散布図の描画 """

    # 20色のカラーマップ
    cm = plt.cm.get_cmap('tab20')

    # クラスごとに可視化
    for label in np.unique(y):
        
        # 8, 10はスキップ
        if label in [8, 10]:
            continue

        plt.scatter(X[y == label, 0], X[y == label, 1], 
                    label=str(label), marker=".", color=cm.colors[label])
        
    plt.legend()
    plt.savefig(path + name + ".png")
    plt.close()

def visualize(X, y):

    path = "Results/matsuda/on/"

    # T-SNE
    # for i in [10, 50, 100]:
    #     tsne = TSNE(n_components=2, random_state=41, perplexity=i)
    #     tsne_transformed = tsne.fit_transform(X)
    #     scatter_plot(tsne_transformed, y, "tsne_" + str(i))

    # PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    pca_transformed = pca.fit_transform(X)
    scatter_plot(pca_transformed, y, path, "pca")

    # UMAP
    # umap = UMAP(n_components=2, random_state=0, n_neighbors=50)
    # umap_X = umap.fit_transform(X)
    # scatter_plot(umap_X, y, "umap")
    
def visualize_with_model(model, X, y):

    path = "Results/L2/"

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval() # 推論モードへ
    X_tensor = torch.from_numpy(X).to(device)

    # 100次元に特徴抽出
    outputs, features = model(X_tensor)
    features = features.to('cpu').detach().numpy().copy()
    outputs = outputs.to('cpu').detach().numpy().copy()

    # T-SNE
    for i in [10, 50, 100]:
        tsne = TSNE(n_components=2, random_state=41, perplexity=i)
        tsne_transformed = tsne.fit_transform(features)

        scatter_plot(tsne_transformed, y, path, "tsne_" + str(i))

    # PCA
    pca = PCA(n_components=2)
    pca.fit(features)
    pca_transformed = pca.fit_transform(features)
    scatter_plot(pca_transformed, y, path, "pca")

    # UMAP
    umap = UMAP(n_components=2, random_state=0, n_neighbors=50)
    umap_X = umap.fit_transform(features)
    scatter_plot(umap_X, y, path, "umap")
    plt.close()

