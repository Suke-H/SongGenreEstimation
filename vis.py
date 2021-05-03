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

    # print(X)
    # print(y)

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

def scatter_plot2(X, y_pred, y_true, path, name):
    """ 散布図の描画 """

    # print(X)
    # print(y)

    # 20色のカラーマップ
    cm = plt.cm.get_cmap('tab20')

    indices_TF = (y_pred == y_true).astype(np.int)
    print(indices_TF)

    # クラスごとに可視化
    for label in np.unique(y_pred):
        
        # 8, 10はスキップ
        # if label in [8, 10]:
        #     continue

        # 正解・不正解で場合分け

        # (色は推定ラベル)
        indices_correct = np.where((y_pred == label) & (indices_TF == 1))[0]
        indices_wrong = np.where((y_pred == label) & (indices_TF == 0))[0]
        # (色は正解ラベル)
        # indices_correct = np.where((y_true == label) & (indices_TF == 1))[0]
        # indices_wrong = np.where((y_true == label) & (indices_TF == 0))[0]

        print("label: {}, correct: {}, wrong: {}".format(label, len(indices_correct), len(indices_wrong)))
        
        plt.scatter(X[indices_correct, 0], X[indices_correct, 1], 
                    label=str(label), marker=".", color=cm.colors[label])

        plt.scatter(X[indices_wrong, 0], X[indices_wrong, 1], 
                    marker="x", color=cm.colors[label])
        
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

def visualize_with_model_list(model_list, X, y, y_true=None):

    path = "Results/L2/"

    # print(X.shape)
    # print(y.shape)
    # print(y_true.shape)

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.from_numpy(X).to(device)

    num_folds = len(model_list)
    feature_list = np.zeros((num_folds, X_tensor.shape[0], 20))

    for i, model in enumerate(model_list):

        model = model.to(device)
        model.eval() # 推論モードへ

        # 20次元の特徴抽出
        _, features = model(X_tensor)
        features = features.to('cpu').detach().numpy().copy()

        feature_list[i] = features

    # 平均
    features = np.mean(feature_list, axis=0)
    print(features.shape)

    # T-SNE
    for i in [10, 50, 100]:
        tsne = TSNE(n_components=2, random_state=41, perplexity=i)
        tsne_transformed = tsne.fit_transform(features)

        if y_true is None:
            scatter_plot(tsne_transformed, y, path, "tsne_" + str(i))

        else:
            scatter_plot2(tsne_transformed, y, y_true, path, "tsne_" + str(i))

    # PCA
    pca = PCA(n_components=2)
    pca.fit(features)
    pca_transformed = pca.fit_transform(features)

    if y_true is None:
        scatter_plot(tsne_transformed, y, path, "pca")

    else:
        scatter_plot2(tsne_transformed, y, y_true, path, "pca")

    # UMAP
    umap = UMAP(n_components=2, random_state=0, n_neighbors=50)
    umap_X = umap.fit_transform(features)

    if y_true is None:
        scatter_plot(tsne_transformed, y, path, "umap")

    else:
        scatter_plot2(tsne_transformed, y, y_true, path, "umap")
