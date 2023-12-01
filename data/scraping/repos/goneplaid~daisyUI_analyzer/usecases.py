"""
This is the template use-case that this repository will analyze scraped data for.
This repository's goal is to analyze the scraped data and to prepare data for the following template.
The following template will be used to generate code but will exist in another repository.
This is only for discovery purposes.
"""

from langchain.llms import OpenAI
from langchain import PromptTemplate

import os

openai = OpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=os.environ['OPENAI_API_KEY']
)

prompt_template = """
You are an expert front end engineer, specializing in building React component libraries with TypeScript (TS) and Tailwind (TW).

Context:

  - Given a spec and some usage examples; you can generate code for a React component, known as the "Primary Component".
  - The primary component is the main component that you generate code for, but it is not the only component that you generate code for.
  - Each component may have a container component and child components, which are also generated.
  - Each component has at least one accompanying classname, used for styling purposes, along with other modifier classnames, which are used for futher custmization of the component.
  - These other classnames will be controlled through props, the public API of the component.
  - All together, these components and classnames are used to compose a component, which is a combination of the primary component, container component, and child components.

A spec consists of the following:

  - Primary component name
  - Primary component description
  - Primary component classname
  - A list of possible container and child components, if any
      - Each container/child component has a name, description, and classname, also
  - A TS component prop interface (the primary component's public API), which is used to control the component's behavior and are each associated with a classname or set of classnames.
      - A mapping of prop names and their associated classnames, known as a "prop-classname mappings".
  - Usage examples are a list of examples that show how the component is used in a real world scenario. They demonstrate the composablity of the component and its consituent component parts and are written in JSX.
      - Some examples may contain other components which aren't part of the spec, but are part of the component library. These components are known as "external library components", for which you do not have to generate anything for.

Given the context and a spec, generate the following primary component:

  - Primary component name: {primary_component_name}
  - Primary component description: {primary_component_description}
  - Primary component className: {primary_component_classname}
  - Container and child components list: {container_and_child_components_list}
  - Primary component TS prop interface: ```
{primary_component_prop_interface}```
  - Prop-classname mappings: ```
{prop_classname_mappings}```
  - Usage examples: ```
{usage_examples}```

Generated code:"""

example_prompt = PromptTemplate(
    input_variables=["primary_component_name",
                     "primary_component_description",
                     "primary_component_classname",
                     "container_and_child_components_list",
                     "primary_component_prop_interface",
                     "prop_classname_mappings",
                     "usage_examples"],
    template=prompt_template
)

# The following code should be supplied to the system as input; this hard-coded example is for proof-of-concept only.

primary_component_name = "Indicator"
primary_component_description = "Indicators are used to place an element on the corner of another element."
primary_component_classname = "indicator"
container_and_child_components_list = """
  - child component:
    - name: IndicatorItem
    - description: will be placed on the corner of sibling
    - className: indicator-item
"""

component_prop_interface = """
interface IndicatorProps {
  horizontalPosition?: 'start' | 'center' | 'end'
  verticalPosition?: 'top' | 'middle' | 'bottom'
}
"""

prop_classname_mappings = """
  - prop: horizontalPosition
    - start: indicator-start
    - center: indicator-center
    - end: indicator-end
  - prop: verticalPosition
    - top: indicator-top
    - middle: indicator-middle
    - bottom: indicator-bottom
"""

usage_examples = """
{/* Example: Empty badge as indicator */}
<Indicator> {/* primary component */}
  <IndicatorItem> {/* child component */}
    <Badge /> {/* external library component */}
  </IndicatorItem> 
  <div>content</div> {/* external library component */}
</Indicator>

{/* Example: Badge as indicator */}
<Indicator> {/* primary component */}
  <IndicatorItem> {/* child component */}
    <Badge>new</Badge> {/* external library component */}
  </IndicatorItem>
  <div>content</div> {/* external library component */}
</Indicator>

{/* Example: for button */}
<Indicator> {/* primary component */}
  <IndicatorItem> {/* child component */}
    <Badge>99+</Badge> {/* external library component */}
  </IndicatorItem>
  <Button>inbox</Button> {/* external library component */}
</Indicator>

{/* Example: for an input */}
<Indicator> {/* primary component */}
  <IndicatorItem> {/* child component */}
    <Badge>Required</Badge> {/* external library component */}
  </IndicatorItem>
  <Input /> {/* external library component */}
</Indicator>
"""

print(
    prompt_template.format(
        primary_component_name=primary_component_name,
        primary_component_description=primary_component_description,
        primary_component_classname=primary_component_classname,
        container_and_child_components_list=container_and_child_components_list,
        primary_component_prop_interface=component_prop_interface,
        prop_classname_mappings=prop_classname_mappings,
        usage_examples=usage_examples,
    )
)

print(
    openai(
        prompt_template.format(
            primary_component_name=primary_component_name,
            primary_component_description=primary_component_description,
            primary_component_classname=primary_component_classname,
            container_and_child_components_list=container_and_child_components_list,
            primary_component_prop_interface=component_prop_interface,
            prop_classname_mappings=prop_classname_mappings,
            usage_examples=usage_examples,
        )
    )
)
