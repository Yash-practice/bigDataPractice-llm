import plotly.graph_objects as go

def plot_gauge(zones, max_value=100, min_value=0):
    # Define the gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge",
        gauge=dict(
            axis=dict(range=[min_value, max_value]),
            bgcolor="lightgray",
            borderwidth=2,
            bordercolor="gray",
            steps=[{'range': zone['range'], 'color': zone['color']} for zone in zones]
        ),
        number=dict(font_size=48, suffix=" units")
    ))
    
    zone_y_axis = 0.5
    # for zone in zones:
    for zone in zones:
        fig.add_annotation(
            dict(
                xref='paper',
                yref='paper',
                x=0.5,
                y=zone_y_axis,
                text=zone['name'],
                showarrow=False,
                font=dict(size=16, color=zone['color']),
                align='center',
                xanchor='center',
                yanchor='middle',
                bgcolor='rgba(255,255,255,0)',
                bordercolor=zone['color'],
                borderwidth=1
            )
        )
        zone_y_axis -= 0.2
    
    # Update layout
    fig.update_layout(
        title_text="Sentiment",
        title_x=0.5,
        width=600,
        height=400
    )

    return fig
