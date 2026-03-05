"""
Plotting utilities
"""
import matplotlib.pyplot as plt


def plot_model_comparison(results: dict, save_path: str = None):
    metrics = ["accuracy", "precision", "recall", "f1_score"]

    dt = [results["decision_tree"][m] for m in metrics]
    lr = [results["logistic_regression"][m] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure()
    plt.bar([i - width/2 for i in x], dt, width, label="Decision Tree")
    plt.bar([i + width/2 for i in x], lr, width, label="Logistic Regression")
    plt.xticks(list(x), ["Acc", "Prec", "Rec", "F1"])
    plt.ylim(0, 1)
    plt.title("Model Metrics Comparison")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path: str = None):
    classes = sorted(set(y_true + y_pred))
    idx = {c: i for i, c in enumerate(classes)}

    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    tick_names = [str(c) for c in classes]
    plt.xticks(range(n), tick_names, rotation=45, ha="right")
    plt.yticks(range(n), tick_names)

    for i in range(n):
        for j in range(n):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f" Saved plot to: {save_path}")

    plt.show()

import matplotlib.pyplot as plt

def plot_decision_tree_manual(root, feature_names=None, class_names=None,
                              save_path: str = None, max_depth: int = None):
    """
    Plot a custom DecisionTree (built with Node/DecisionTree classes)
    using a nicer automatic layout.

    Parameters
    ----------
    root : Node
        Root node of the decision tree (e.g. your_tree.root)
    feature_names : list of str, optional
        Names of features. If None, will use f0, f1, f2, ...
    class_names : list of str, optional
        Mapping from class index/value to label for display.
    save_path : str, optional
        If set, save the figure to this path.
    max_depth : int, optional
        If set, only draw nodes up to this depth (useful if the tree is huge).
    """



    def is_leaf(node):

        return node.is_leaf()

    def tree_depth(node):
        if node is None or is_leaf(node):
            return 0
        return 1 + max(tree_depth(node.left), tree_depth(node.right))



    def node_label(node):
        if is_leaf(node):
            if class_names is not None:
                try:
                    cls = class_names[node.value]
                except Exception:
                    cls = str(node.value)
            else:
                cls = str(node.value)
            return f"class = {cls}"

        if feature_names is not None and node.feature is not None:
            feat_name = feature_names[node.feature]
        else:
            feat_name = f"f{node.feature}"

        return f"{feat_name} <= {node.threshold:.3f}"




    leaves_inorder = []

    def collect_leaves(node):
        if node is None:
            return
        if not is_leaf(node):
            collect_leaves(node.left)
            collect_leaves(node.right)
        else:
            leaves_inorder.append(node)

    collect_leaves(root)
    n_leaves = max(1, len(leaves_inorder))


    leaf_x = {}
    if n_leaves == 1:
        leaf_x[leaves_inorder[0]] = 0.5
    else:
        for i, leaf in enumerate(leaves_inorder):
            leaf_x[leaf] = 0.1 + 0.8 * i / (n_leaves - 1)


    depth_cache = {}

    def compute_depths(node, d=0):
        if node is None:
            return
        depth_cache[node] = d
        if not is_leaf(node):
            compute_depths(node.left, d + 1)
            compute_depths(node.right, d + 1)

    compute_depths(root, 0)
    max_tree_depth = max(depth_cache.values()) if depth_cache else 0


    pos = {}

    def assign_x(node):
        if node is None:
            return
        if node in leaf_x:
            pos[node] = [leaf_x[node], None]
        else:
            assign_x(node.left)
            assign_x(node.right)
            xs = []
            if node.left in pos:
                xs.append(pos[node.left][0])
            if node.right in pos:
                xs.append(pos[node.right][0])
            if xs:
                pos[node] = [sum(xs) / len(xs), None]
            else:
                pos[node] = [0.5, None]

    assign_x(root)


    for node, d in depth_cache.items():
        if max_tree_depth == 0:
            y = 0.5
        else:
            y = 1.0 - 0.9 * (d / max_tree_depth)
        pos[node][1] = y



    fig_w = max(8, n_leaves * 0.8)
    fig_h = max(4, (max_tree_depth + 1) * 1.2)
    plt.figure(figsize=(fig_w, fig_h))

    def draw_node(node):
        if node is None:
            return

        d = depth_cache[node]
        if max_depth is not None and d > max_depth:
            return

        x, y = pos[node]
        label = node_label(node)
        leaf = is_leaf(node)
        facecolor = "#90EE90" if leaf else "#ADD8E6"

        plt.text(
            x, y, label,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc=facecolor, ec="black"),
            fontsize=8,
        )


        if not leaf:
            for child in [node.left, node.right]:
                if child is None:
                    continue
                if max_depth is not None and depth_cache[child] > max_depth:
                    continue
                cx, cy = pos[child]
                plt.plot([x, cx], [y - 0.01, cy + 0.01], "k-", linewidth=0.8)
                draw_node(child)

    draw_node(root)

    plt.axis("off")
    plt.title("Decision Tree (Custom Implementation)")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f" Saved decision tree plot to: {save_path}")

    plt.show()
