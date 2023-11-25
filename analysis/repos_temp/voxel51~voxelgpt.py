"""
Query: {query}
Available runs: {available_runs}
Selected run: {selected_run}\n
""""""
No evaluation runs found. If you want to evaluate your predictions (`pred_field`) against ground truth labels (`gt_field`), run the appropriate evaluation method:

```py
# ex: detection
dataset.evaluate_detections(pred_field, gt_field=gt_field, eval_key="eval")

# ex: classification
dataset.evaluate_classifications(pred_field, gt_field=gt_field, eval_key="eval")
```
""""""
No uniqueness runs found. If you want to compute uniqueness, run the following command:

```py
import fiftyone.brain as fob

fob.compute_uniqueness(dataset)
```
""""""
No mistakenness runs found. To compute the difficulty of classifying samples (`pred_field`) with respect to ground truth labels (`gt_field`), run the following command:

```py
import fiftyone.brain as fob

fob.compute_mistakenness(
    dataset,
    pred_field,
    label_field=gt_field,
)
```
""""""
No similarity index found. To generate a similarity index for your samples, run the following command:

```py
import fiftyone.brain as fob

fob.compute_similarity(dataset, brain_key="img_sim")
```
""""""
No similarity index found that supports text prompts. To generate a similarity index for your samples, run the following command:

```py
import fiftyone.brain as fob

fob.compute_similarity(
    dataset,
    model="clip-vit-base32-torch",
    brain_key="text_sim",
)
```
""""""
No hardness run found. To measure of the uncertainty of your model's predictions (`label_field`) on the samples in your dataset, run the following command:

```py
import fiftyone.brain as fob

fob.compute_hardness(dataset, label_field)
```
""""""
No metadata found. To compute metadata for your samples, run the following command:

```py
dataset.compute_metadata()
```
""""""
    Query: {query}
    Available fields: {available_fields}
    Required fields: {required_fields}\n
    """"""
    Query: {query}
    Label field: {field}
    Classes: {label_classes}\n
    """"""
    Class name: {class_name}
    Available label classes: {available_label_classes}
    Semantic matches: {semantic_matches}\n
    """"""
        Query: {query}
        Algorithms used: {algorithms}\n
        """"""
    Candidate tag: {candidate_tag}
    Allowed tags: {allowed_tags}
    Selected tags: {selected_tags}\n
    """"""
    Query: {query}
    Is history relevant: {history_is_relevant}
    """"""
    Query: {query}
    Intent: {intent}\n
    """"""
    View stage: {view_stage}
    Description: {description}
    Inputs: {inputs}\n
    """"""
A uniqueness run determines how unique each image is in the dataset. Its results are stored in the {uniqueness_field} field on the samples.
When converting a natural language query into a DatasetView, if you determine that the uniqueness of the images is important, a view stage should use the {uniqueness_field} field.
""""""
A hardness run scores each image based on how difficult it is to classify for a specified label field. In this task, the hardness of each sample for the {label_field} field is has been scored, and its results are stored in the {hardness_field} field on the samples.
""""""
An image_similarity run determines determines how similar each image is to another image. You can use the {image_similarity_key} key to access the results of this run and sort images by similarity.
""""""
A text_similarity run determines determines how similar each image is to a user-specified input text prompt. You can use the {text_similarity_key} key to access the results of this run and find images that most resemble the description in the user-input text prompt. You can use these and only these brian_key values brain_key="{brain_key}" for an output using sort_by_similarity.
""""""
A mistakenness run determines how mistaken each image is in the dataset. Its results are stored in the {mistakenness_field} field on the samples.
When converting a natural language query into a DatasetView, if you determine that the mistakenness of the images is important, the following fields store relevant information:
- {mistakenness_field}: the mistakenness score for each image
""""""
An evaluation run computes metrics, statistics, and reports assessing the accuracy of model predictions for classifications, detections, and segmentations. You can use the {eval_key} key to access the results of this run, including TP, FP, and FNs.
"""