import editdistance_s


class Matcher:
    def __init__(self):
        self.vocabs = {}
        self.poi_slots = {'poi名称', 'poi修饰', 'poi目标', '起点名称', '起点修饰', '起点目标', '终点名称', '终点修饰', '终点目标', '途经点名称'}

        with open('./data/lexicon/poi_name.txt', 'r', encoding='utf-8') as file:
            for line in file.readlines():
                word = line.strip()
                if len(word) not in self.vocabs:
                    self.vocabs[len(word)] = []
                self.vocabs[len(word)].append(word)

    def match(self, slot, value):
        if slot.split('-')[1] not in self.poi_slots:
            return value
        if value in self.vocabs.get(len(value), set()):
            return value
        if len(value) <= 4:
            return value

        min_dist, min_val = 1000, ''

        words = self.vocabs.get(len(value), []) + self.vocabs.get(len(value) - 1, []) + self.vocabs.get(len(value) + 1, [])
        for w in words:
            dist = editdistance_s.distance(value, w)
            if dist < min_dist:
                min_dist = dist
                min_val = w

        return min_val if min_dist <= 1 else value
