def plot_loss_phyl(losses_dict, distances_dict, plot_filename):
    # Plot loss
    legend_info = []
    plt.title("Loss")
    for lang_a in losses_dict:
        for lang_b in losses_dict[lang_a]:
            losses = losses_dict[lang_a][lang_b]
            loss_x = [p[0] for p in losses]
            loss_y = [p[1] for p in losses]
            loss_line, = plt.plot(loss_x, loss_y, label=lang_a + "-" + lang_b)
            legend_info.append(loss_line)
    plt.legend(handles=legend_info)
    plt.savefig(plot_filename + "_loss.png")
    plt.close()
    
    # Plot loss
    legend_info = []
    plt.title("Edit distance")
    for lang_a in distances_dict:
        for lang_b in distances_dict[lang_a]:
            distances = distances_dict[lang_a][lang_b]
            dist_x = [p[0] for p in distances]
            dist_y = [p[1] for p in distances]
            distance_line, = plt.plot(dist_x, dist_y, label=lang_a + "-" + lang_b)
            legend_info.append(distance_line)
    plt.legend(handles=legend_info)
    plt.savefig(plot_filename + "_dist.png")
    plt.close()