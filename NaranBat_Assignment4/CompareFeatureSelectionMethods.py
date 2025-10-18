# CompareFeatureSelectionMethods.py

# chatGPT assisted in getting the correct imports and function definitions
import numpy as np
import pandas as pd
import random
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score


# chatGPT assisted in defining helper functions
def two_fold_cv_decision_tree(X, y, random_state=1):
    # Perform 2-fold stratified cross-validation with a decision tree classifier
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state) # ensure reproducibility
    cms = []
    accs = []
    # For each fold
    for train_idx, test_idx in skf.split(X, y):
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        cms.append(confusion_matrix(y[test_idx], preds))
        accs.append(accuracy_score(y[test_idx], preds))
    agg_cm = sum(cms)
    mean_acc = float(np.mean(accs))
    return agg_cm, mean_acc

def pretty_print_cm(cm):
    # Pretty print confusion matrix
    rows = []
    for r in cm:
        rows.append("[ " + "  ".join(f"{int(x):3d}" for x in r) + " ]")
    return "\n".join(rows)



# Part 1

def part1(X, y, feature_names):
    # Evaluate using all original features
    print("\n Part 1: Original 4 features ")
    cm, acc = two_fold_cv_decision_tree(X, y, random_state=1)
    print("Confusion matrix:")
    print(pretty_print_cm(cm))
    print(f"Accuracy: {acc:.4f}")
    print(f"Features used: {feature_names}")
    return cm, acc


# Part 2 - PCA

# chatGPT assisted in debugging PCA part and printing results
def part2_pca(X, y):
    # PCA transformation and selection based on PoV
    pca = PCA(n_components=4)
    Z = pca.fit_transform(X)
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_.T
    pov = pca.explained_variance_ratio_

    print("\n Part 2: PCA transform (z1..z4) ")
    print("Eigenvalues (lambda_i):")
    for i, val in enumerate(eigenvalues, start=1):
        print(f"  z{i}: {val:.6f}")

    print("\nEigenvectors matrix (rows = original features; columns = components z1..z4):")
    for i, row in enumerate(eigenvectors, start=1):
        print(f"  orig_feat_{i}: " + ", ".join(f"{v:+0.6f}" for v in row))

    print("\nPoV (proportion of variance) per component:")
    for i, val in enumerate(pov, start=1):
        print(f"  z{i}: {val:.6f}")
    cum_pov = np.cumsum(pov)
    print("Cumulative PoV:", cum_pov)

    # Select enough components for PoV > 0.90
    k = np.argmax(cum_pov >= 0.90) + 1
    selected_idx = list(range(k))
    selected_features = [f"z{i+1}" for i in selected_idx]
    print(f"Selected components (cumulative PoV > 0.90): {selected_features} (k = {k})")

    # Evaluate decision tree on selected PCA features
    X_sel = Z[:, selected_idx]
    cm, acc = two_fold_cv_decision_tree(X_sel, y, random_state=1)
    print("\nPart 2 Confusion matrix (using selected z components):")
    print(pretty_print_cm(cm))
    print(f"Part 2 Accuracy: {acc:.4f}")
    return Z, eigenvalues, eigenvectors, pov, selected_idx, cm, acc



# Part 3 - Simulated Annealing

def part3_simulated_annealing(all_X, y, all_feature_names, iterations=100, restart=10, c=1.0, seed=1):
    # Simulated Annealing for feature selection
    print("\n Part 3: Simulated Annealing ")
    random.seed(seed)
    np.random.seed(seed)
    n = all_X.shape[1]  # 8 features

    # Initialize with a random non-empty mask
    mask = [random.choice([True, False]) for _ in range(n)]
    if not any(mask):
        mask[random.randrange(n)] = True
    best_mask = mask[:]
    best_score, _ = evaluate_subset_mask(best_mask, all_X, y)

    print(f"Initial subset: {[all_feature_names[i] for i,v in enumerate(best_mask) if v]} | Accuracy: {best_score:.4f}")

    for it in range(1, iterations+1):
        # perturb 1 or 2 parameters (flip)
        k = random.choice([1, 2])
        new_mask = best_mask[:]
        idxs = random.sample(range(n), k)
        for idx in idxs:
            new_mask[idx] = not new_mask[idx]

        # Evaluate
        score, _ = evaluate_subset_mask(new_mask, all_X, y)
        delta = score - best_score

        # Temperature schedule: linear cooling T in (1 -> ~0)
        T = max(0.001, 1.0 - (it / iterations))

        if delta > 0:
            pr_accept = 1.0
            u = None
            status = "Improved"
            best_mask = new_mask[:]
            best_score = score
        else:
            # acceptance probability
            pr_accept = float(np.exp((delta * c) / T))
            u = random.random()
            if u < pr_accept:
                status = "Accepted"
                best_mask = new_mask[:]
                best_score = score
            else:
                status = "Discarded"

        subset_names = [all_feature_names[i] for i, v in enumerate(new_mask) if v]
     
        # Print iteration info
        ru_str = '-' if u is None else f"{u:.6f}"
        print(f"Iter {it:03d}: Subset={subset_names} | Accuracy={score:.4f} | Pr[accept]={pr_accept:.6f} | RandomUniform={ru_str} | Status={status}")

        # chatGPT assisted in fixing the restart logic
        # Restart logic
        if it % restart == 0:
            rand_mask = [random.choice([True, False]) for _ in range(n)]
            if not any(rand_mask):
                rand_mask[random.randrange(n)] = True
            rand_score, _ = evaluate_subset_mask(rand_mask, all_X, y)
            print(f"*** Restart at iter {it}: New random subset={[all_feature_names[i] for i,v in enumerate(rand_mask) if v]} | Accuracy={rand_score:.4f} ***")
            if rand_score > best_score:
                best_mask = rand_mask[:]
                best_score = rand_score

    # After all iterations, report best found
    final_selected = [all_feature_names[i] for i, v in enumerate(best_mask) if v]
    final_cm, final_acc = two_fold_cv_decision_tree(all_X[:, best_mask], y, random_state=1)
    print("\nSimulated Annealing final selected features:", final_selected)
    print("Confusion matrix:")
    print(pretty_print_cm(final_cm))
    print(f"Accuracy: {final_acc:.4f}")
    return best_mask, best_score, final_cm, final_acc

def evaluate_subset_mask(mask, all_X, y):
    # Evaluate a feature subset given by mask
    if not any(mask):
        return 0.0, np.zeros((3, 3), dtype=int)
    X_sub = all_X[:, mask]
    cm, acc = two_fold_cv_decision_tree(X_sub, y, random_state=1)
    return acc, cm


# Part 4 - Genetic Algorithm

def part4_genetic_algorithm(all_X, y, all_feature_names, initial_sets, generations=50, pop_size=20, seed=1):
    # Genetic Algorithm for feature selection
    print("\n Part 4: Genetic Algorithm ")
    random.seed(seed)
    np.random.seed(seed)
    name_to_idx = {name: i for i, name in enumerate(all_feature_names)}

    # Convert feature names to mask
    def names_to_mask(names):
        m = [False] * len(all_feature_names)
        for n in names:
            if n in name_to_idx:
                m[name_to_idx[n]] = True
        if not any(m):
            m[random.randrange(len(m))] = True
        return m
    
    #chatGPT assisted in fixing the population logic
    # initial population: convert provided sets, then pad with random masks
    population = [names_to_mask(s) for s in initial_sets]
    while len(population) < pop_size:
        m = [random.choice([True, False]) for _ in range(len(all_feature_names))]
        if any(m):
            population.append(m)

    def fitness(mask):
        # fitness = accuracy from 2-fold CV
        score, _ = evaluate_subset_mask(mask, all_X, y)
        return score

    def mask_to_str(mask):
        # convert mask to feature names string
        return ", ".join([all_feature_names[i] for i, v in enumerate(mask) if v])

    def crossover(a, b):
        # one-point crossover
        point = random.randint(1, len(a) - 1)
        child = a[:point] + b[point:]
        if not any(child):
            child[random.randrange(len(child))] = True
        return child

    def mutate(mask, p=0.1):
        new = mask[:]
        for i in range(len(new)):
            if random.random() < p:
                new[i] = not new[i]
        if not any(new):
            new[random.randrange(len(new))] = True
        return new

    for gen in range(1, generations + 1):
        scored = [(fitness(ind), ind) for ind in population]
        scored.sort(key=lambda x: x[0], reverse=True)
        top5 = scored[:5]
        print(f"\nGeneration {gen} top 5:")
        for rank, (scr, mask) in enumerate(top5, start=1):
            print(f"  {rank}. Features=[{mask_to_str(mask)}] | Accuracy={scr:.4f}")

        # selection: roulette wheel (with small epsilon to avoid zero-sum)
        eps = 1e-8
        fitnesses = [s for s, _ in scored]
        total = sum(fitnesses) + eps * len(fitnesses)
        if total == 0:
            # if all zero, choose uniformly
            probs = [1.0 / len(scored)] * len(scored)
        else:
            probs = [(s + eps) / total for s in fitnesses]

        # produce next generation
        next_pop = []
        while len(next_pop) < pop_size:
            parents_idx = np.random.choice(range(len(scored)), size=2, replace=False, p=probs)
            parent_a = scored[parents_idx[0]][1]
            parent_b = scored[parents_idx[1]][1]
            child = crossover(parent_a, parent_b)
            child = mutate(child, p=0.1)
            next_pop.append(child)
        population = next_pop

    # after final generation, report best
    final_scored = [(fitness(ind), ind) for ind in population]
    final_scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_mask = final_scored[0]
    best_selected = [all_feature_names[i] for i, v in enumerate(best_mask) if v]
    final_cm, final_acc = two_fold_cv_decision_tree(all_X[:, best_mask], y, random_state=1)
    print("\nGenetic Algorithm final selected features:", best_selected)
    print("Confusion matrix:")
    print(pretty_print_cm(final_cm))
    print(f"Accuracy: {final_acc:.4f}")
    return best_mask, best_score, final_cm, final_acc


# Main 
def main():
    # Load Iris
    iris = datasets.load_iris()
    X = iris.data.copy()  # columns: sepal-length, sepal-width, petal-length, petal-width
    y = iris.target.copy()
    base_feature_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']

    # Part 1
    cm1, acc1 = part1(X, y, base_feature_names)

    # Part 2
    Z, eigenvalues, eigenvectors, pov, selected_idx, cm2, acc2 = part2_pca(X, y)
    z_names = [f"z{i+1}" for i in range(Z.shape[1])]

    # Build combined feature matrix (original 4 + z1..z4)
    all_feature_names = base_feature_names + z_names
    all_X = np.hstack([X, Z])  # shape (n_samples, 8)

    # Part 3 - Simulated Annealing
    sa_mask, sa_score, cm3, acc3 = part3_simulated_annealing(all_X, y, all_feature_names,
                                                             iterations=100, restart=10, c=1.0, seed=1)

    # Part 4 - Genetic Algorithm
    initial_sets = [
        ['z1', 'sepal-length', 'sepal-width', 'petal-length', 'petal-width'],
        ['z1', 'z2', 'sepal-width', 'petal-length', 'petal-width'],
        ['z1', 'z2', 'z3', 'sepal-width', 'petal-length'],
        ['z1', 'z2', 'z3', 'z4', 'sepal-width'],
        ['z1', 'z2', 'z3', 'z4', 'sepal-length']
    ]
    ga_mask, ga_score, cm4, acc4 = part4_genetic_algorithm(all_X, y, all_feature_names, initial_sets,
                                                          generations=50, pop_size=20, seed=1)

    # Summary of final accuracies and features
    print("\n Final Summary ")
    print(f"Part 1 Accuracy (original 4 features): {acc1:.4f}")
    print(f"Part 2 Accuracy (PCA-selected features): {acc2:.4f}")
    print(f"Part 3 Accuracy (Simulated Annealing selected): {acc3:.4f}")
    print(f"Part 4 Accuracy (Genetic Algorithm selected): {acc4:.4f}")

    # Print final feature lists for parts
    print("\nFinal feature sets used:")
    print(f"  Part 1: {base_feature_names}")
    z_selected_names = [f"z{i+1}" for i in selected_idx]
    print(f"  Part 2: {z_selected_names}")
    sa_selected = [all_feature_names[i] for i, v in enumerate(sa_mask) if v]
    print(f"  Part 3: {sa_selected}")
    ga_selected = [all_feature_names[i] for i, v in enumerate(ga_mask) if v]
    print(f"  Part 4: {ga_selected}")

if __name__ == "__main__":
    main()
