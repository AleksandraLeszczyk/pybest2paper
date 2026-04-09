import logging
import os
from typing import Any, Literal, Union

import numpy as np
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
import pandas as pd
import plotly.express as px


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
  answer with bool returned by tool.
"""


@tool
def create_interactive_plot(
    data: dict[str, list[Union[str, float]]],
    plot_type: Literal["line", "bar", "scatter", "histogram", "box", "area"] = "line",
    x: str = None,
    y: str | list[str] = None,
    color: str = None,
    title: str = "Interactive Plot",
) -> bool:
    """
    Creates an interactive Plotly figure from a pandas DataFrame.

    Args:
        data:      dictionary with keys being columns names and values being column values
        plot_type: One of 'line', 'bar', 'scatter', 'histogram', 'box', 'area'.
        x:         Column name for the x-axis (uses index if None).
        y:         Column name(s) for the y-axis (uses all numeric cols if None).
        color:     Column name used to color-code the series.
        title:     Chart title.

    Returns:
        True if success.
    """
    try:
        df = pd.DataFrame(data)

        # Fall back to numeric columns when y is not specified
        if y is None:
            y = df.select_dtypes(include="number").columns.tolist()

        # Use index as x-axis if not specified
        if x is None:
            df = df.copy()
            df["__index__"] = df.index
            x = "__index__"

        plot_fn = {
            "line": px.line,
            "bar": px.bar,
            "scatter": px.scatter,
            "histogram": px.histogram,
            "box": px.box,
            "area": px.area,
        }

        if plot_type not in plot_fn:
            raise ValueError(
                f"Unsupported plot_type '{plot_type}'. " f"Choose from: {list(plot_fn)}"
            )

        kwargs = dict(data_frame=df, x=x, title=title)

        # histogram only accepts a single column for x — skip y/color
        if plot_type == "histogram":
            kwargs["x"] = y[0] if isinstance(y, list) else y
            if color:
                kwargs["color"] = color
        else:
            kwargs["y"] = y
            if color:
                kwargs["color"] = color

        fig = plot_fn[plot_type](**kwargs)

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        # Save to artifacts
        figures = [i for i in os.listdir("artifacts") if i.startswith("fig") and i.endswith("png")]
        fig.write_html(f"artifacts/fig{len(figures)}.html")
        return True
    
    except Exception:
        return False

@tool
def create_molecule_plot(
    basis: Any,
    matrix_ao_mo: np.ndarray,
    index: int,
    isovalue: float = 0.07,
    title: str = "Molecule"
):
    raise NotImplementedError

model_viz_creator = ChatOpenAI(temperature=0, model_name=MODEL)

viz_creator = create_agent(
    model_viz_creator,
    tools=[create_interactive_plot],
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": PROMPT_VIZ_CREATOR,
            }
        ]
    ),
)
