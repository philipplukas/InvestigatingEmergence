import os 

SPLITS = {"train": "train.txt",
          "test": "test.txt",
          "valid": "valid.txt"}

BASE_DIR = os.path.dirname(__file__)

class CTLDataStatistics():

    def get_stastics(split: str) -> dict:
        path = os.path.join(BASE_DIR, "ctl", SPLITS[split])

        stats = {}
        total = 0

        with open(path, mode='r') as fp:
            for line in fp.readlines():
                line = line.rstrip()

                # Exclude first two characters since they are the start of sequence token ^ and the input digit
                function_part = line.split(":")[0][2:]
                depth = len(function_part)

                if depth in stats:
                    stats[depth] += 1
                else:
                    stats[depth] = 1
                total += 1
        
        for key, value in stats.items():
            stats[key] = value / total

        return stats


CTLDataStatistics.get_stastics = staticmethod(CTLDataStatistics.get_stastics)
get_data_statistics = CTLDataStatistics.get_stastics