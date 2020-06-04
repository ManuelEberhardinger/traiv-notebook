from torchvision import transforms
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import os
import json
import logging

root = logging.getLogger()

class TransformationHelper:

    def __init__(self):
        self.std_image_size = 224
        self.std_scale = 256

    def get_trainings_transformations(self):
        return transforms.Compose([
                #transforms.Resize(self.std_scale),
                #transforms.Lambda(lambda x: x.convert('L')),
                transforms.RandomResizedCrop(self.std_image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def get_test_transformations(self):
        return transforms.Compose([
                #transforms.Lambda(lambda x: x.convert('L')),
                transforms.Resize(self.std_scale),
                transforms.CenterCrop(self.std_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    # for doublets and small images
    def change_std_img_and_scale(self):
        self.std_image_size =  112
        self.std_scale = 128

    # for the normal images and the predefined models
    def reset_to_std_img_and_scale(self):
        self.std_image_size = 224
        self.std_scale = 256


class TestHelper:

    def __init__(self, data_loader, dset_classes, net, device):
        self.data_loader = data_loader
        self.classes = dset_classes
        self.size = len(dset_classes)
        self.net = net.to(device)
        self.class_correct = list(0. for i in range(self.size))
        self.class_total = list(0. for i in range(self.size))
        self.device = device

    # test the total precision for two classes
    def test_total_precision(self):
        total = correct = 0
        for data in self.data_loader:
            (images, labels_data), (path, _) = data
            labels = (Variable(labels_data)).to(self.device)
            outputs = self.net((Variable(images)).to(self.device))
            probs = F.softmax(outputs, 1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            c = (predicted == labels).squeeze()
            label = labels[0]
            self.class_correct[label] += c.item()
            self.class_total[label] += 1
            probs = probs.data.cpu().numpy()[0]
            percentage_probs = [i * 100 for i in probs]
            predicted = predicted.cpu().numpy()[0]
            root.debug(path[0], "Predicted: ", self.classes[predicted])
            root.debug(self.classes)
            root.debug(percentage_probs)
            root.debug()
            labels = labels.cpu().numpy()[0]
            #if (predicted != labels):
                # print the false classified test data
                #root.debug(path[0] + ': ' + self.classes[predicted] + " " + str(probs[0]) + " " + str(probs[1]))
                #shutil.copy2(path[0], false_pred + path[0].split("/")[-1][:-4] + "_" + self.dset_classes[predicted] + ".png")

    # print the statistics of the classes, prints also the sensitivity and specificity
    def print_total_precision(self, name, epoch):
        with open('results.txt', 'a') as f:
            root.debug(name, "Epoch:", epoch, file=f)
            for i in range(self.size):
                if(self.class_total[i] == 0):
                    continue
                root.debug('Accuracy of %5s : %2d %%' % (
                    self.classes[i], 100 * self.class_correct[i] / self.class_total[i]), file=f)
                root.debug(self.classes[i] + ": " + str(self.class_correct[i]) + " of " + str(self.class_total[i]) + " images", file=f)

            root.debug("Total images: " + str(len(self.data_loader)), file=f)
            correct = 0
            for i in range(len(self.class_correct)):
                correct = correct + self.class_correct[i]

            root.debug("Total precision: %2d %%" % (100*correct / sum(self.class_total)), file=f)


class JsonLocationsHelper:

    def __init__(self, path):
        self.path = path
        self.classes = [x for x in os.listdir(path) if "." not in x]
        self.locations = None
        self.country_mapping = self.load_country_mapping()
        self.all_image_infos = {}
        self.load_all_locations()
        root.debug(str(self.classes))

    def load_country_mapping(self):
        file_path = os.path.join(self.path, "country-alpha-2-mapping.json")

        with open(file_path, "r") as f:
            content = json.load(f)

        root.debug("loaded {} country to alpha 2 code mappings.".format(len(content)))
        return content

    def load_all_locations(self):
        if self.locations is None:
            self.locations = {}
            for label in self.classes:
                self.locations[label.lower()] = self.load_locations_for_label(label)
        root.debug("loaded {} labels".format(len(self.all_image_infos)))
        return self.locations

    def load_locations_for_label(self, label):
        root.debug("load_locations_for_label > label: {}".format(label))
        label_path = os.path.join(self.path, label)
        locations = {}
        files = [x for x in os.listdir(label_path) if x.endswith("json")]
        i = 0
        for file in files:
            json_file_path = os.path.join(label_path, file)
            with open(json_file_path, "r") as f:
                image_id = os.path.splitext(file)[0]
                content = json.load(f)
                if content is not None:
                    country = content["attributes"]["table"]["country"]
                    id_infos = content["attributes"]["table"]
                    id_infos['id'] = image_id
                    if country is not None:

                        # get alpha 2 code
                        alpha_code = self.country_mapping.get(country.strip(), country.strip())
                        id_infos['country'] = alpha_code
                        if alpha_code not in locations:
                            locations[alpha_code] = []

                        i = i + 1
                        locations[alpha_code].append(id_infos)
                    self.all_image_infos[image_id] = id_infos
        root.debug("load_locations_for_label > found {} locations".format(i))
        return locations

    def get_locations_for_label(self, label):
        if self.locations is None:
            self.load_all_locations()

        locations = self.locations.get(label.lower())
        return locations

    def get_all_labels(self):
        return self.classes

    def get_information_for_id(self, id):
        return self.all_image_infos[id]

    def get_all_ids(self):
        return list(self.all_image_infos.keys())
