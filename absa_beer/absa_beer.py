"""
Main module for executing tasks of methodology

:author: Denis Eiras

Functions:
    - read_data: Read data from file
    - select_complete_rows: Select complete rows
    - generate_descriptive_statistics: Generate statistics

"""

from step_1 import Step_1
from step_2 import Step_2
from step_3 import Step_3


# Main function
def main():

    run_step_1 = False
    if run_step_1:
        step_1 = Step_1()
        # testing 2 pages and 2 reviews
        step_1.max_beer_page = 2
        step_1.max_page_reviews = 2
        step_1.run()

    step_2 = Step_2()
    step_2.run()
    step_2.generate_descriptive_statistics("step_2_stats.csv")

    step_3 = Step_3()
    step_3.run()
    step_3.generate_descriptive_statistics("step_3_stats.csv")


if __name__ == "__main__":
    main()
