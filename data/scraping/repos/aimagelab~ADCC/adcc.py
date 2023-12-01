from metrics import coherency, complexity, average_drop
import torch
import torch.nn.functional as FF

def ADCC(image, saliency_map, explanation_map,arch,attr_method,target_class_idx=None, debug=False):
    if torch.cuda.is_available():
        image = image.cuda()
        explanation_map=explanation_map.cuda()
        arch = arch.cuda()

    with torch.no_grad():
        out = FF.softmax(arch(image), dim=1)

    avgdrop = average_drop.average_drop(image, explanation_map, arch, out=out, class_idx=target_class_idx)
    coh,A,B=coherency.coherency(saliency_map,explanation_map, arch=arch, attr_method=attr_method, out=out)
    com=complexity.complexity(saliency_map)


    adcc = 3 / (1/coh + 1/(1-com) +1/(1-avgdrop))

    if debug:
        return adcc, avgdrop, coh, com, A, B
    return adcc