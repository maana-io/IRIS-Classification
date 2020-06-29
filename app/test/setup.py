
import random as rd
from app.logic.helpers import id


words = '''
        The distribution of oil and gas reserves among the world's 50 largest oil companies. The reserves of the privately
        owned companies are grouped together. The oil produced by the "supermajor" companies accounts for less than 15% of
        the total world supply. Over 80% of the world's reserves of oil and natural gas are controlled by national oil companies.
        Of the world's 20 largest oil companies, 15 are state-owned oil companies.
        The petroleum industry, also known as the oil industry or the oil patch, includes the global processes of exploration,
        extraction, refining, transporting (often by oil tankers and pipelines), and marketing of petroleum products.
        The largest volume products of the industry are fuel oil and gasoline (petrol). Petroleum (oil) is also the raw material
        for many chemical products, including pharmaceuticals, solvents, fertilizers, pesticides, synthetic fragrances, and plastics.
        The industry is usually divided into three major components: upstream, midstream and downstream. Midstream operations are
        often included in the downstream category.
        Petroleum is vital to many industries, and is of importance to the maintenance of industrial civilization in its
        current configuration, and thus is a critical concern for many nations. Oil accounts for a large percentage of the
        world’s energy consumption, ranging from a low of 32% for Europe and Asia, to a high of 53% for the Middle East.
        Governments such as the United States government provide a heavy public subsidy to petroleum companies, with major
        tax breaks at virtually every stage of oil exploration and extraction, including the costs of oil field leases and
        drilling equipment.[2]
        Principle is a term defined current-day by Merriam-Webster[5] as: "a comprehensive and fundamental law, doctrine,
        or assumption", "a primary source", "the laws or facts of nature underlying the working of an artificial device",
        "an ingredient (such as a chemical) that exhibits or imparts a characteristic quality".[6]
        Process is a term defined current-day by the United States Patent Laws (United States Code Title 34 - Patents)[7]
        published by the United States Patent and Trade Office (USPTO)[8] as follows: "The term 'process' means process,
        art, or method, and includes a new use of a known process, machine, manufacture, composition of matter, or material."[9]
        Application of Science is a term defined current-day by the United States' National Academies of Sciences, Engineering,
        and Medicine[12] as: "...any use of scientific knowledge for a specific purpose, whether to do more science; to design
        a product, process, or medical treatment; to develop a new technology; or to predict the impacts of human actions."[13]
        The simplest form of technology is the development and use of basic tools. The prehistoric discovery of how to control
        fire and the later Neolithic Revolution increased the available sources of food, and the invention of the wheel
        helped humans to travel in and control their environment. Developments in historic times, including the printing
        press, the telephone, and the Internet, have lessened physical barriers to communication and allowed humans to
        interact freely on a global scale.
        Technology has many effects. It has helped develop more advanced economies (including today's global economy)
        and has allowed the rise of a leisure class. Many technological processes produce unwanted by-products known as
        pollution and deplete natural resources to the detriment of Earth's environment. Innovations have always influenced
        the values of a society and raised new questions of the ethics of technology. Examples include the rise of the
        notion of efficiency in terms of human productivity, and the challenges of bioethics.
        Philosophical debates have arisen over the use of technology, with disagreements over whether technology improves
        the human condition or worsens it. Neo-Luddism, anarcho-primitivism, and similar reactionary movements criticize
        the pervasiveness of technology, arguing that it harms the environment and alienates people; proponents of ideologies
        such as transhumanism and techno-progressivism view continued technological progress as beneficial to society and
        the human condition.
        Health care or healthcare is the maintenance or improvement of health via the prevention, diagnosis, and treatment
        of disease, illness, injury, and other physical and mental impairments in human beings. Healthcare is delivered by
        health professionals (providers or practitioners) in allied health fields. Physicians and physician associates are
        a part of these health professionals. Dentistry, midwifery, nursing, medicine, optometry, audiology, pharmacy,
        psychology, occupational therapy, physical therapy and other health professions are all part of healthcare. It
        includes work done in providing primary care, secondary care, and tertiary care, as well as in public health.
        Access to health care may vary across countries, communities, and individuals, largely influenced by social and
        economic conditions as well as the health policies in place. Countries and jurisdictions have different policies
        and plans in relation to the personal and population-based health care goals within their societies. Healthcare
        systems are organizations established to meet the health needs of targeted populations. Their exact configuration
        varies between national and subnational entities. In some countries and jurisdictions, health care planning is
        distributed among market participants, whereas in others, planning occurs more centrally among governments or
        other coordinating bodies. In all cases, according to the World Health Organization (WHO), a well-functioning
        healthcare system requires a robust financing mechanism; a well-trained and adequately paid workforce; reliable
        information on which to base decisions and policies; and well maintained health facilities and logistics to deliver
        quality medicines and technologies.[1]
        Health care is conventionally regarded as an important determinant in promoting the general physical and mental
        health and well-being of people around the world. An example of this was the worldwide eradication of smallpox
        in 1980, declared by the WHO as the first disease in human history to be completely eliminated by deliberate health
        care interventions.[4]
        '''.split()


def random_feature():
    feature_types = ['NUMERICAL', 'CATEGORICAL', 'TEXT', 'SET', 'BOOLEAN']
    fIndex = rd.randint(0, 100)
    fName = 'feature' + str(fIndex) + id()[:8]
    fTypeIdx = rd.randint(0, len(feature_types)-1)
    fType = feature_types[fTypeIdx]

    return {
        'id': id(),
        'index': fIndex,
        'name': fName,
        'type': fType
    }


def random_train_group_feature():
    fIndex = rd.randint(0, 100)
    fName = 'TRAIN_GROUP'
    fType = 'NUMERICAL'

    return {
        'id': id(),
        'index': fIndex,
        'name': fName,
        'type': fType
    }


def random_dataentry(ftype):
    if ftype in ["CATEGORICAL"]:
        num_categories = rd.randint(6, 22)
        dval = "string " + str(rd.randint(1, num_categories))
        return {'id': id(), 'text': dval}
    elif ftype in ["TEXT"]:
        random_words = []
        for _ in range(rd.randint(50, 300)):
            random_words.append(words[rd.randint(0, len(words)-1)])
        dval = " ".join(random_words)
        return {'id': id(), 'text': dval}
    elif ftype in ["NUMERICAL"]:
        dval = rd.randint(0, 100)*1.0/100
        return {'id': id(), 'numerical': dval}
    elif ftype in ["BOOLEAN"]:
        dval = rd.randint(0, 1)
        return {'id': id(), 'numerical': dval}
    else: #SET
        num_categories = rd.randint(3, 10)
        dval = []
        for _ in range(rd.randint(1, 3)):
            dval.append("string " + str(rd.randint(1, num_categories)))
        return {'id': id(), 'set': dval}



def random_labeldata(num_classes):
    dval = "class " + str(rd.randint(1, num_classes))
    return {'id': id(), 'text': dval}



def dataentry_value(data, ftype):
    if ftype in ["CATEGORICAL", "TEXT"]:
        return data['text']
    elif ftype in ["NUMERICAL"]:
        return data['numerical']
    elif ftype in ["BOOLEAN"]:
        return data['numerical']
    else: #SET
        return data['set']


def random_dataset():
    num_features = rd.randint(10, 20)
    num_entries = rd.randint(100, 200)
    feature_data = []

    feature = random_train_group_feature()
    fvalues = []
    for _ in range(num_entries):
        fvalues.append({'id': id(), 'numerical': rd.randint(1,5)})
    feature_data.append({
        'id': id(),
        'feature': feature,
        'data': fvalues
    })

    for _ in range(num_features):
        feature = random_feature()
        fvalues = []
        for _ in range(num_entries):
            fvalues.append(random_dataentry(feature['type']))
        feature_data.append({
            'id': id(),
            'feature': feature,
            'data': fvalues
        })

    return {'id': id(), 'features': feature_data}



def random_labeled_dataset(num_classes=rd.randint(6, 8)):
    dataset = random_dataset()
    num_features = len(dataset['features'])
    label = {
        'id': id(),
        'index': num_features,
        'name': "Label",
        'type': "LABEL"
    }
    data = []
    num_entries = len(dataset['features'][0]['data'])

    for _ in range(num_entries):
        data.append(random_labeldata(num_classes=num_classes))
    feature_data = {'id': id(), 'feature': label, 'data': data}

    #Must be given unique ID.
    return {'id': id(),
            'data': dataset,
            'label': feature_data}



def random_class_weights(num_features, labels):
    class_weights = []
    for label in labels:
        intercept = rd.randint(0, 100)*1.0/100
        feature_weights = []
        for _ in range(num_features):
            num_weights = rd.randint(1, 10)
            feature_weights.append({'id': id(), 'feature': random_feature(), 'weights': [rd.randint(0, 100)*1.0/100] * num_weights })
        class_weights.append({'id': id(), 'weights': feature_weights, 'class': label, 'intercept': intercept})

    return class_weights



def random_noop():
    return {'id': id()}



def random_min_max_scaler():
    return {
            'id': id(),
            'minValue': rd.random(),
            'maxValue': rd.randint(1, 10) + rd.random(),
            'scale': rd.random(),
            'dataMin': rd.random(),
            'dataMax': rd.randint(1, 10) + rd.random()
    }



def random_label_binarizer(num_classes):
    return {
            'id': id(),
            'labels': [ "Label " + str(n+1) for n in range(num_classes)]
    }



def random_multilabel_binarizer(num_classes):
    return  {
            'id': id(),
            'labels': [ "Label " + str(n+1) for n in range(num_classes)]
    }



def random_label_encoder(num_classes):
    return {
            'id': id(),
            'labels': [ "Label " + str(n+1) for n in range(num_classes)]
    }



def random_tfidf_vectorizer():
    num_terms = rd.randint(100, 600)
    term_feature_mapping = []
    idfs = []
    for tidx in range(num_terms):
        term = "term" + str(tidx)
        fidx = 100 + tidx
        term_feature_mapping.append({'id': id(), 'term': term, 'featureIdx': fidx})
        idfs.append(rd.random())

    return {
            'id': id(),
            'vocab': term_feature_mapping,
            'idf': idfs,
            'stopwords': ['the', 'this', 'a', 'an', 'those', 'these', 'at', 'on']
    }



def random_doc_to_vector():
    return {
            'id': id(),
            'modelFile': 'fullpathText2VecBinaryFileName',
            'maxNumWords': rd.randint(1000, 10000)
    }



def random_model_performance(num_classes):
    class_performances = []
    for lblidx in range(num_classes):
        label = "Label " + str(lblidx+1)
        buckets = []
        total_num_instances = 0
        for bucket_idx in range(num_classes):
            num_instances = rd.randint(20, 80)
            total_num_instances += num_instances
            buckets.append({
                'id': id(),
                'trueLabel': label,
                'predictedLabel': "Label " + str(bucket_idx+1),
                'numInstances': num_instances,
                'weight': rd.random(),
            })
        perf = {
            'id': id(),
            'label': label,
            'weight': rd.random(),
            'numInstances': total_num_instances,
            'classifiedAs': buckets,
            'recall': rd.random(),
            'precision': rd.random(),
            'f1': rd.random()
        }
        class_performances.append(perf)

    return {
        'id': id(),
        'classPerformances': class_performances,
        'numInstances': rd.randint(50, 100),
        'avgRecall': rd.random(),
        'avgPrecision': rd.random(),
        'avgF1': rd.random()
    }



def random_batch_classification_results():
    dataset = random_dataset()
    num_classes = rd.randint(2, 5)
    probabilities = [1.0/num_classes] * num_classes
    classes = ["Class " + str(idx+1) for idx in range(num_classes)]

    allPredictedLabels = [{
        'id': id(),
        'label': lbl,
        'probability': prob
    } for (lbl, prob) in zip(classes, probabilities)]

    class_summaries = []
    for clsidx in range(num_classes):
        num_instances = rd.randint(3, 10)
        results = []
        for instidx in range(num_instances):
            data_idx = rd.randint(0, len(dataset['features'][0]['data'])-1)
            input_data = []
            for feat in dataset['features']:
                input_data.append({'id': id(), 'feature': feat['feature'], 'data': [feat['data'][data_idx]]})
            data_instance = {'id': id(), 'features': input_data}
            results.append({
                'id': id(),
                'dataInstance': {'id': id(), 'dataset': data_instance, 'index': instidx},
                'allLabels': allPredictedLabels,
                'predictedLabel': allPredictedLabels[clsidx],
                'entropy': rd.random(),
                'contributors': [{'id': id(), 'featureName': 'topology', 'featureValue': 'topsides', 'weight': .68}],
                'recommends': [{'id': id(), 'featureName': 'topology', 'featureValue': 'subsea', 'weight': .86}]
            })
        class_summaries.append({
            'id': id(),
            'label': classes[clsidx],
            'numInstances': num_instances,
            'probabilities': [1.0/num_classes] * num_instances,
            'entropies': [rd.random()] * num_instances,
            'results': results
        })

    return {
        'id': id(),
        'classSummaries': class_summaries
    }