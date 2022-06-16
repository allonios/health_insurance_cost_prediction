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

def plot_grid_search(plt, cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.xlabel(name_param_1, fontsize=16)
    ax.ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')