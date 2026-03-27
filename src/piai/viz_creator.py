import logging
import os
from typing import Union

import seaborn as sns
import matplotlib.pyplot as plt
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage 
import pandas as pd


logger = logging.getLogger()

# In a real environment, this URL would point to your actual MCP server
MODEL = os.environ["MODEL_VIZ_CREATOR"]


PROMPT_VIZ_CREATOR = """
You are data analyst with expertise in vizualization and quantum chemistry.
You create interactive plots with description using tools.
Then tools don't return image directly but save them in the registry.
You don't access registry but get update if figure creation succeeded.
If there is an error message, try to fix it.

Example 
question: Plot potential energy surface x=1.6, 1.7, 1.8, e=-99, -99.1, -99
your steps:
  create_interactive_plot(
    data = \{'r': [1.6, 1.7, 1.8], 'energy': [-99, -99.1, -99]\},
    x='r',
    y='energy',
    title='Potential energy surface',
    description='Potential energy surface.'
    )
  answer 'The figure with potential energy surface has been created and saved in registry.'

Your answer should only include status update, e.g.
'The requested figure has been created and saved in registry.'
or 'I am unable to create this figure because <...>'
"""

@tool
def create_plot(
    data: dict[str, list[Union[str, float]]],
    plot_type: Union[list[str], str] = "line",
    x: str = None,
    y: Union[str, list[str]] = None,
    color: str = None,
    title: str = "Figure",
    description: str = ""
):
    """
    Generates a Seaborn plot based on the specified parameters.
    """
    sns.set(font_scale=2)
    # Convert dict to DataFrame
    df = pd.DataFrame(data)
    
    # Handle plot_type if it's passed as a list (taking the first element)
    if isinstance(plot_type, list):
        plot_type = plot_type[0]
    
    # Set the visual style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Logic for handling multiple Y columns
    # If Y is a list, we melt the dataframe to make it "long-form" for Seaborn
    if isinstance(y, list) and len(y) > 1:
        id_vars = [x] if x else []
        if color and color not in y:
            id_vars.append(color)
            
        df = df.melt(id_vars=id_vars, value_vars=y, var_name="Variable", value_name="Value")
        y_col = "Value"
        hue_col = "Variable" if color is None else color
    else:
        y_col = y if isinstance(y, str) else (y[0] if y else None)
        hue_col = color

    # Plot Dispatcher
    try:
        if plot_type == "line":
            ax = sns.lineplot(data=df, x=x, y=y_col, hue=hue_col)
        elif plot_type == "bar":
            ax = sns.barplot(data=df, x=x, y=y_col, hue=hue_col)
        elif plot_type == "scatter":
            ax = sns.scatterplot(data=df, x=x, y=y_col, hue=hue_col)
        elif plot_type == "histogram":
            ax = sns.histplot(data=df, x=x or y_col, hue=hue_col, kde=True)
        elif plot_type == "box":
            ax = sns.boxplot(data=df, x=x, y=y_col, hue=hue_col)
        elif plot_type == "area":
            ax = sns.lineplot(data=df, x=x, y=y_col, hue=hue_col)
            plt.fill_between(df[x], df[y_col], alpha=0.3)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        # Formatting
        plt.title(title, fontsize=15, pad=20)
        plt.xlabel(x if x else "")
        plt.ylabel(y_col if isinstance(y_col, str) else "")
        
        # Display description below the plot
        if description:
            plt.figtext(0.5, -0.05, description, wrap=True, horizontalalignment='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        fig = ax.get_figure()
        figures = [i for i in os.listdir("artifacts") if i.startswith("fig") and i.endswith("png")]
        fig.savefig(f"artifacts/fig{len(figures)}.png")
        return f"Success. Saved figure as artifacts/fig{len(figures)}.png"
        
    except Exception as e:
        return f"An error occurred while plotting: {e}"


model_viz_creator = ChatOpenAI(temperature=0, model_name=MODEL)

viz_creator = create_agent(
    model_viz_creator,
    tools=[create_plot],
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": PROMPT_VIZ_CREATOR,
            }
        ]
    ),
)
