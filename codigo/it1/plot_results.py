import json
import sys
import matplotlib.pyplot as plt


def plot_results(stats1, stats2, stats3):
    training_episodes = stats1["training_stats_episodes"]
    if 'training_losses' in stats1:  # DQN
        training_losses = (stats1["training_losses"], stats2["training_losses"], stats3["training_losses"])
    else:
        training_losses = ((stats1["training_value_losses"], stats1["training_action_losses"]),
                           (stats2["training_value_losses"], stats2["training_action_losses"]),
                           (stats3["training_value_losses"], stats3["training_action_losses"]))

    plot_length = stats1["validation_episodes"][-1]+1

    # Plot 1: training loss
    plt.figure(figsize=(16, 8))
    plt.title('Error de entrenamiento')
    if 'training_losses' in stats1:
        plt.plot(training_episodes, training_losses[0], 'b', label='atari')
        plt.plot(training_episodes, training_losses[1], 'r', label='alexnet')
        plt.plot(training_episodes, training_losses[2], 'g', label='mobilenet')
        plt.ylim([5, 40])
    else:
        plt.plot(training_episodes, training_losses[0][0], 'b', label='atari valor')
        plt.plot(training_episodes, training_losses[0][1], 'b.--', label='atari acción')
        plt.plot(training_episodes, training_losses[1][0], 'r', label='alexnet valor')
        plt.plot(training_episodes, training_losses[1][1], 'r.--', label='alexnet acción')
        plt.plot(training_episodes, training_losses[2][0], 'g', label='mobilenet valor')
        plt.plot(training_episodes, training_losses[2][1], 'g.--', label='mobilenet acción')
        plt.ylim([-5, 420])
    plt.legend(loc='upper right')
    plt.ylabel('Error de entrenamiento')
    plt.xlabel('Episodio')
    plt.xlim([0, plot_length])
    plt.show()

    training_accuracy = (stats1["training_hits"],stats2["training_hits"],stats3["training_hits"])
    validation_episodes=stats1["validation_episodes"]

    validation_accuracy=(stats1["validation_hits"],stats2["validation_hits"],stats3["validation_hits"])
    validation_episodes_for_accuracy=validation_episodes.copy()
    validation_episodes_for_accuracy.insert(0,0)
    validation_accuracy[0].insert(0,0.7)
    validation_accuracy[1].insert(0, 0.7)
    validation_accuracy[2].insert(0, 0.7)

    # Plot 2: top 1 accuracy
    plt.figure(figsize=(16, 8))
    plt.title('Precision (accuracy)')
    plt.plot(training_episodes, training_accuracy[0],'b', label='atari entrenamiento')
    plt.plot(training_episodes, training_accuracy[1], 'r',label='alexnet entrenamiento')
    plt.plot(training_episodes, training_accuracy[2],'g', label='mobilenet entrenamiento')
    plt.plot(validation_episodes_for_accuracy, validation_accuracy[0], 'bo--', label='atari validación')
    plt.plot(validation_episodes_for_accuracy, validation_accuracy[1], 'ro--', label='alexnet validación')
    plt.plot(validation_episodes_for_accuracy, validation_accuracy[2], 'go--', label='mobilenet validación')
    plt.legend(loc='upper right')
    plt.ylabel('Porcentaje de acierto')
    plt.ylim([0.7, 0.95])
    plt.xlim([0, plot_length])
    plt.xlabel('Episodio')
    plt.show()

    validation_class_changes = ([x[0] for x in stats1["validation_class_changes"]],
                                [x[0] for x in stats2["validation_class_changes"]],
                                [x[0] for x in stats3["validation_class_changes"]])
    validation_class_better = ([x[1] for x in stats1["validation_class_changes"]],
                               [x[1] for x in stats2["validation_class_changes"]],
                               [x[1] for x in stats3["validation_class_changes"]])
    validation_class_worse = ([x[2] for x in stats1["validation_class_changes"]],
                              [x[2] for x in stats2["validation_class_changes"]],
                              [x[2] for x in stats3["validation_class_changes"]])

    # plot 3: cambios de clase (en validacion)
    plt.figure(figsize=(16, 8))
    plt.title('Cambios de clase en validación')

    plt.plot(validation_episodes, validation_class_changes[0], 'b-', label='cambios atari')
    plt.plot(validation_episodes, validation_class_changes[1], 'r-', label='cambios alexnet')
    plt.plot(validation_episodes, validation_class_changes[2], 'g-', label='cambios mobilenet')

    plt.plot(validation_episodes, validation_class_better[0], 'b--', label='mejora atari')
    plt.plot(validation_episodes, validation_class_better[1], 'r--', label='mejora alexnet')
    plt.plot(validation_episodes, validation_class_better[2], 'g--', label='mejora mobilenet')

    plt.plot(validation_episodes, validation_class_worse[0], 'bo-', label='empeora atari')
    plt.plot(validation_episodes, validation_class_worse[1], 'ro-', label='empeora alexnet')
    plt.plot(validation_episodes, validation_class_worse[2], 'go-', label='empeora mobilenet')


    plt.legend(loc='upper right')
    plt.ylabel('Porcentaje de cambios de clase')
    plt.ylim([0, 0.25])
    plt.xlim([1000, plot_length])
    plt.xlabel('Episodio')
    plt.show()


    #plot 4: Recompensas en validacion
    validation_rewards = (stats1["validation_rewards"],
                          stats2["validation_rewards"],
                          stats3["validation_rewards"])
    plt.figure(figsize=(16, 8))
    plt.title('Recompensas en validación')
    plt.plot(validation_episodes, validation_rewards[0], 'bo--', label='Atari')
    plt.plot(validation_episodes, validation_rewards[0], 'ro--', label='Alexnet')
    plt.plot(validation_episodes, validation_rewards[0], 'go--', label='Mobilenet')

    plt.legend(loc='upper right')
    plt.ylabel('Recompensa (diferencia entre valor inicial e final * 10')
    plt.ylim([0, 0.25])
    plt.xlim([1000, plot_length])
    plt.xlabel('Episodio')
    plt.show()

    #bar 1: Histograma de cambios de clase por tipo de cambio
    plt.figure(figsize=(16, 8))
    plt.title('Cambios de clase en validación')
    final_validation_class_changes = [validation_class_changes[0][-1],
                                      validation_class_changes[1][-1],
                                      validation_class_changes[2][-1],
                                      validation_class_better[0][-1],
                                     validation_class_better[1][-1],
                                     validation_class_better[2][-1],
                                    validation_class_worse[0][-1],
                                    validation_class_worse[1][-1],
                                    validation_class_worse[2][-1]]

    names=['cambios atari','cambios alexnet','cambios mobilenet','mejora atari','mejora alexnet','mejora mobilenet',
           'empeora atari','empeora alexnet','empeora mobilenet']
    plt.bar(names,final_validation_class_changes)
    plt.show()

    # bar 2: Histograma de cambios de clase por modelo
    plt.figure(figsize=(16, 8))
    plt.title('Cambios de clase en validación')
    final_validation_class_changes = [validation_class_changes[0][-1],
                                      validation_class_better[0][-1],
                                      validation_class_worse[0][-1],
                                      validation_class_changes[1][-1],
                                      validation_class_better[1][-1],
                                      validation_class_worse[1][-1],
                                      validation_class_changes[2][-1],
                                      validation_class_better[2][-1],
                                      validation_class_worse[2][-1],
                                      ]

    names = ['cambios atari',  'mejora atari', 'empeora atari',
             'cambios alexnet',  'mejora alexnet', 'empeora alexnet',
             'cambios mobilenet','mejora mobilenet', 'empeora mobilenet']
    plt.bar(names, final_validation_class_changes)
    plt.show()

    # bar 3: Histograma de recompensas por modelo

    final_validation_rewards = [stats1["validation_rewards"][-1],
        stats2["validation_rewards"][-1],
        stats3["validation_rewards"][-1],
        stats1["validation_positive_rewards"][-1],
        stats2["validation_positive_rewards"][-1],
        stats3["validation_positive_rewards"][-1],
        ]

    plt.figure(figsize=(16, 8))
    plt.title('Cambios de clase en validación')

    names = ['recompensa atari',
             'recompensa alexnet',
             'recompensa mobilenet',
             '% recompensa positiva atari',
             '% recompensa positiva alexnet',
             '% recompensa positiva mobilenet',
             ]
    plt.bar(names, final_validation_rewards)
    plt.show()


if len(sys.argv) != 4:
    print("Uso :  python plot_results.py fichero_log1.json fichero_log2.json fichero_log3.json")
else:
    with open(sys.argv[1], "r") as read_file:
        stats1 = json.load(read_file)
    with open(sys.argv[2], "r") as read_file:
        stats2 = json.load(read_file)
    with open(sys.argv[3], "r") as read_file:
        stats3 = json.load(read_file)
    plot_results(stats1, stats2, stats3)
