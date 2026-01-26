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
from step_4 import Step_4
from step_5 import Step_5


# Main function
def main():

    step_1 = Step_1()
    step_1.run()

    step_2 = Step_2()
    step_2.run()
    step_2.generate_descriptive_statistics("step_2_stats.csv")

    step_3 = Step_3()
    step_3.run()
    step_3.generate_descriptive_statistics("step_3_stats.csv")
    
    step_4 = Step_4()
    step_4.run()
    step_4.generate_descriptive_statistics("step_4_stats.csv")

    step_5 = Step_5()
    step_5.run()

if __name__ == "__main__":
    main()
