import matplotlib.pyplot as plt

def plot_clusters(df):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df['avg_x'], df['avg_y'], c=df['pattern_cluster'])
    plt.xlabel("Average X Position")
    plt.ylabel("Average Y Position")
    plt.title("Attacking Pattern Clusters")
    plt.colorbar(scatter)
    plt.show()
