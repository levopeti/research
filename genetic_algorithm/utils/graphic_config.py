from graphics import GraphWin, Entry, Point, Text, Rectangle
import yaml
from threading import Thread
import methaboard

config = None
mb = None


def load_logs(config_file):
    global config
    with open(config_file, 'r') as config_file:
        config = yaml.load(config_file)


def write_logs(config_file):
    global config
    with open(config_file, "w+") as log_file:
        yaml.dump(config, log_file, default_flow_style=False)


def button(ll_point, ur_point, text, color="white"):
    but = Rectangle(Point(ll_point[0], ll_point[1]), Point(ur_point[0], ur_point[1]))  # points are ordered ll, ur
    but.setFill(color)

    text = Text(Point((ll_point[0] + ur_point[0]) // 2, (ll_point[1] + ur_point[1]) // 2), text)
    but.draw(win)
    text.draw(win)

    return but, text


def push_button(name, but):
    if config[name]:
        config[name] = False
        but[0].setFill("red")
    else:
        config[name] = True
        but[0].setFill("green")


def modify_button(but):
    global selection_types, mutation_types
    name = but[1].getText()

    if name in selection_types:
        index = selection_types.index(name)
        new_index = (index + 1) % len(selection_types)
        but[1].setText(selection_types[new_index])
        config["selection_type"] = selection_types[new_index]

    if name in mutation_types:
        index = mutation_types.index(name)
        new_index = (index + 1) % len(mutation_types)
        but[1].setText(mutation_types[new_index])
        config["mutation_type"] = mutation_types[new_index]


def init_color(name):
    if config[name]:
        return "green"
    else:
        return "red"


def make_text(x, y, size, text):
    text = Text(Point(x, y), text)
    text.setSize(size)
    text.draw(win)
    return text


def make_text_with_input(x, y, col, size, text, init_value="", width=10):
    begin = 0
    if col == 1:
        begin = 220
    elif col == 2:
        begin = 620
    text = make_text(x, y, size, text)
    entry = Entry(Point(begin, y), width)
    entry.setFill("white")
    entry.setText(init_value if init_value != "None" else "")
    entry.draw(win)
    return entry, text


def inside(point, button):
    """ Is point inside rectangle? """
    rectangle = button[0]

    ll = rectangle.getP1()  # assume p1 is ll (lower left)
    ur = rectangle.getP2()  # assume p2 is ur (upper right)

    return ll.getX() < point.getX() < ur.getX() and ll.getY() > point.getY() > ur.getY()


def refresh_config():
    for entry in entries:
        name = entry[1].getText()
        if name == "num of new indiv.:":
            name = "num_of_new_individual"
        else:
            name = name.replace(' ', '_')
            name = name[:-1]

        if name in ["CR", "F", "sigma_init", "sigma_fin", "mutation_probability", "step_size"]:
            config[name] = None if entry[0].getText() == '' else float(entry[0].getText())
        else:
            config[name] = None if entry[0].getText() == '' else int(entry[0].getText())

    write_logs("/home/biot/projects/research/genetic_algorithm/config_tmp.yml")


def run_methaboard():
    global mb

    if mb is None:
        mb = methaboard.MethaBoard()
        path = methaboard_entry[0].getText()
        thread = Thread(target=mb.run, args=(path, ))
        thread.setDaemon(False)
        thread.start()

        methaboard_button[0].undraw()
        methaboard_button[1].undraw()
        methaboard_entry[0].undraw()
        methaboard_entry[1].undraw()


load_logs("/home/biot/projects/research/genetic_algorithm/config.yml")
write_logs("/home/biot/projects/research/genetic_algorithm/config_tmp.yml")

selection_types = ["random", "tournament", "better half"]
mutation_types = ["basic", "bacterial"]

entries = []

win = GraphWin("MethaConfig", 1400, 900)

active_button = button((950, 300), (1200, 150), "ACTIVE", init_color("active"))
stop_button = button((950, 550), (1200, 450), "STOP", init_color("stop"))
exit_button = button((1150, 870), (1350, 800), "EXIT", "blue")

make_text(110, 20, 16, "Population and other")

population_size_entry = make_text_with_input(75, 50, 1, 14, "population size:", str(config["population_size"]))
chromosome_size_entry = make_text_with_input(88, 80, 1, 14, "chromosome size:", str(config["chromosome_size"]))
pool_button = button((12, 128), (132, 100), "pool", init_color("pool"))
pool_size_entry = make_text_with_input(51, 145, 1, 14, "pool size:", str(config["pool_size"]))
entries += [population_size_entry, chromosome_size_entry, pool_size_entry]

make_text(85, 193, 16, "Stop conditions")

max_iteration_entry = make_text_with_input(68, 223, 1, 14, "max iteration:", str(config["max_iteration"]))
max_fitness_eval_entry = make_text_with_input(81, 253, 1, 14, "max fitness eval:", str(config["max_fitness_eval"]))
min_fitness_entry = make_text_with_input(61, 283, 1, 14, "min fitness:", str(config["min_fitness"]))
patience_entry = make_text_with_input(52, 313, 1, 14, "patience:", str(config["patience"]))
entries += [max_iteration_entry, max_fitness_eval_entry, min_fitness_entry, patience_entry]

make_text(145, 350, 16, "Add new individual methods")

crossover_button = button((12, 397), (132, 369), "crossover", init_color("crossover"))
selection_type_button = button((12, 434), (132, 404), str(config["selection_type"]), "white")
num_of_crossover_entry = make_text_with_input(85, 444, 1, 14, "num of crossover:", str(config["num_of_crossover"]))

diff_evol_button = button((12, 498), (132, 470), "diff. evolution", init_color("differential_evolution"))
CR_entry = make_text_with_input(28, 516, 1, 14, "CR:", str(config["CR"]))
F_entry = make_text_with_input(22, 546, 1, 14, "F:", str(config["F"]))
entries += [num_of_crossover_entry, CR_entry, F_entry]

invasive_weed_button = button((12, 600), (132, 572), "invasive weed", init_color("invasive_weed"))
iter_max_entry = make_text_with_input(50, 618, 1, 14, "iter max:", str(config["iter_max"]))
e_entry = make_text_with_input(23, 648, 1, 14, "e:", str(config["e"]))
sigma_init_entry = make_text_with_input(56, 678, 1, 14, "sigma init:", str(config["sigma_init"]))
sigma_fin_entry = make_text_with_input(55, 708, 1, 14, "sigma fin:", str(config["sigma_fin"]))
N_min_entry = make_text_with_input(42, 738, 1, 14, "N min:", str(config["N_min"]))
N_max_entry = make_text_with_input(45, 768, 1, 14, "N max:", str(config["N_max"]))
entries += [iter_max_entry, e_entry, sigma_init_entry, sigma_fin_entry, N_min_entry, N_max_entry]

add_new_button = button((12, 822), (132, 794), "add pure new", init_color("add_pure_new"))
num_of_new_entry = make_text_with_input(90, 840, 1, 14, "num of new indiv.:", str(config["num_of_new_individual"]))

make_text(545, 350, 16, "Modify all individuals methods")

mutation_button = button((403, 397), (523, 369), "mutation", init_color("mutation"))
mutation_type_button = button((403, 434), (523, 404), str(config["mutation_type"]), "white")
elitism_button = button((403, 468), (465, 440), "elitism", init_color("elitism"))
mrs_button = button((403, 506), (465, 478), "mrs", init_color("mutation_random_sequence"))
mutation_prob_entry = make_text_with_input(485, 525, 2, 14, "mutation probability:", str(config["mutation_probability"]))
num_of_clones_entry = make_text_with_input(463, 555, 2, 14, "num of clones:", str(config["num_of_clones"]))
entries += [num_of_new_entry, mutation_prob_entry, num_of_clones_entry]

memetic_button = button((403, 609), (523, 581), "memetic", init_color("memetic"))
lrs_button = button((403, 647), (465, 619), "lrs", init_color("lamarck_random_sequence"))
step_size_entry = make_text_with_input(442, 666, 2, 14, "step size:", str(config["step_size"]))
number_of_steps_entry = make_text_with_input(471, 696, 2, 14, "number of steps:", str(config["number_of_steps"]))
entries += [step_size_entry, number_of_steps_entry]

# methaboard
methaboard_button = button((403, 822), (520, 794), "run methaboard", "yellow")
methaboard_entry = make_text_with_input(438, 840, 2, 14, "log path:", "/home/biot/projects/research/logs/log", width=30)

while True:
    if config["active"]:
        refresh_config()

    clickPoint = win.getMouse()

    if inside(clickPoint, stop_button):
        push_button("stop", stop_button)
        continue
    if inside(clickPoint, exit_button):
        del mb
        win.close()
        exit()
    if inside(clickPoint, active_button):
        push_button("active", active_button)
        continue
    if inside(clickPoint, pool_button):
        push_button("pool", pool_button)
        continue
    if inside(clickPoint, crossover_button):
        push_button("crossover", crossover_button)
        continue
    if inside(clickPoint, diff_evol_button):
        push_button("differential_evolution", diff_evol_button)
        continue
    if inside(clickPoint, invasive_weed_button):
        push_button("invasive_weed", invasive_weed_button)
        continue
    if inside(clickPoint, add_new_button):
        push_button("add_pure_new", add_new_button)
        continue
    if inside(clickPoint, mutation_button):
        push_button("mutation", mutation_button)
        continue
    if inside(clickPoint, elitism_button):
        push_button("elitism", elitism_button)
        continue
    if inside(clickPoint, mrs_button):
        push_button("mutation_random_sequence", mrs_button)
        continue
    if inside(clickPoint, memetic_button):
        push_button("memetic", memetic_button)
        continue
    if inside(clickPoint, lrs_button):
        push_button("lamarck_random_sequence", lrs_button)
        continue

    if inside(clickPoint, selection_type_button):
        modify_button(selection_type_button)
        continue
    if inside(clickPoint, mutation_type_button):
        modify_button(mutation_type_button)
        continue

    if inside(clickPoint, methaboard_button):
        run_methaboard()
        continue

