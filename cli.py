import argparse

# command line interface

available_commands = ['voc']
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, help=f"one of {available_commands}")
    args = parser.parse_args()
    command = args.command

    assert command in available_commands, f"command must be one of {available_commands}"
    if command == 'voc':
      from segmentation_datasets.voc import generate_voc_masks
      generate_voc_masks()
  
if __name__ == '__main__':
    main()