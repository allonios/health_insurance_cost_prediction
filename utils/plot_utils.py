import numpy as np

def plot_validation_curve(plt, param_range,train_score,val_score):
    print("Validation Curve Data", train_score, val_score, param_range)
    plt.plot(
        param_range,
        np.median(train_score, 1),
        color="blue",
        alpha=0.3,
        linestyle="dashed"
    )
    plt.plot(
        param_range,
        np.median(val_score, 1),
        color="red",
        alpha=0.3,
        linestyle="dashed"
    )
    plt.legend(loc="best")
    plt.title("Validation Curve")
    plt.xlabel("param")
    plt.ylabel("score")
    
def plot_learning_curve(plt, N, train_lc, val_lc):
    plt.plot(
        N,
        np.mean(train_lc, 1),
        color="blue",
        label="training score"
    )
    plt.plot(
        N,
        np.mean(val_lc, 1),
        color="red",
        label="validation score"
    )
    plt.hlines(
        np.mean([
            train_lc[-1],
            val_lc[-1]
        ]),
        N[0],
        N[-1],
        color="gray",
        linestyle="dashed"
    )

    
    plt.xlabel("training size")
    plt.ylabel("score")
    plt.title("Learning Curve")
    plt.legend(loc="best")

def plot_grid_search_results(plt, cv_results, best_params, param_grid):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = cv_results
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(best_params.keys())
    for p_k, p_v in best_params.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()