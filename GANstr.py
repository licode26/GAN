import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from matplotlib.colors import LinearSegmentedColormap
import sys
import os

# Add current directory to path to import from the code
sys.path.append(os.path.dirname(__file__))

from GAN import AdvancedAfforestationPlanningSystem, EnhancedAfforestationDataset, ResidualBlock,EnhancedGenerator, EnhancedDiscriminator, AdvancedGeneticOptimizer, AdvancedGeneticOptimizer


# Set page configuration
st.set_page_config(
    page_title="Advanced Afforestation Planning System",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #388E3C;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #F1F8E9;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1B5E20;
    }
    .metric-label {
        font-size: 1rem;
        color: #558B2F;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_or_train_model(data, feature_columns, progress_bar=None):
    """Load an existing model or train a new one"""
    system = AdvancedAfforestationPlanningSystem(
        feature_dim=len(feature_columns),
        noise_dim=64,
        tree_types=5
    )
    
    # Check for existing model
    if os.path.exists('afforestation_models.pt'):
        try:
            system.load_pretrained_model('afforestation_models.pt')
            st.success("✅ Loaded existing model successfully!")
            return system
        except Exception as e:
            st.warning(f"Could not load existing model: {str(e)}. Training a new one.")
    
    # Train new model
    epochs = 50
    batch_size = 32
    validation_split = 0.2
    
    if progress_bar:
        for i in range(epochs):
            # Update progress bar
            progress_bar.progress((i + 1) / epochs)
            if i == 0:
                system.train(
                    data=data,
                    feature_columns=feature_columns,
                    epochs=1,
                    batch_size=batch_size,
                    validation_split=validation_split
                )
            else:
                # Continue training
                system.train(
                    data=data,
                    feature_columns=feature_columns,
                    epochs=1,
                    batch_size=batch_size,
                    validation_split=validation_split
                )
                
    else:
        system.train(
            data=data,
            feature_columns=feature_columns,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
    
    st.success("✅ Model training completed!")
    return system

def generate_synthetic_data(num_samples=500):
    """Generate synthetic afforestation data"""
    np.random.seed(42)
    data = pd.DataFrame({
        'elevation': np.random.uniform(0, 2000, num_samples),
        'slope': np.random.uniform(0, 45, num_samples),
        'precipitation': np.random.uniform(500, 2000, num_samples),
        'temperature': np.random.uniform(5, 30, num_samples),
        'soil_quality': np.random.uniform(1, 10, num_samples),
        'soil_type': np.random.randint(0, 10, num_samples),
        'climate_zone': np.random.randint(0, 8, num_samples),
        'current_vegetation': np.random.uniform(0, 100, num_samples)
    })

    # Calculate synthetic carbon sequestration
    data['carbon_sequestration'] = (
        0.2 * data['elevation'] / 1000 +
        -0.5 * data['slope'] / 45 +
        0.8 * data['precipitation'] / 2000 +
        -0.3 * (data['temperature'] - 15) / 15 +
        0.6 * data['soil_quality'] / 10 +
        0.4 * data['current_vegetation'] / 100
    ) * 15

    # Add random noise
    data['carbon_sequestration'] += np.random.normal(0, 1, num_samples)
    data['carbon_sequestration'] = np.abs(data['carbon_sequestration'])
    
    return data

def plot_3d_pareto_front(solutions):
    """Create a 3D interactive plot of the Pareto front"""
    carbon_values = [s['carbon_sequestration'] for s in solutions]
    biodiversity_values = [s['biodiversity_index'] for s in solutions]
    cost_values = [s['implementation_cost'] for s in solutions]
    
    # Find the best solutions
    best_carbon_idx = carbon_values.index(max(carbon_values))
    best_biodiversity_idx = biodiversity_values.index(max(biodiversity_values))
    best_cost_idx = cost_values.index(min(cost_values))
    
    # Create a 3D scatter plot
    fig = go.Figure()
    
    # Add all solutions
    fig.add_trace(go.Scatter3d(
        x=carbon_values,
        y=biodiversity_values,
        z=cost_values,
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.7
        ),
        name='Pareto Solutions'
    ))
    
    # Add best carbon solution
    fig.add_trace(go.Scatter3d(
        x=[carbon_values[best_carbon_idx]],
        y=[biodiversity_values[best_carbon_idx]],
        z=[cost_values[best_carbon_idx]],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='diamond'
        ),
        name='Best Carbon'
    ))
    
    # Add best biodiversity solution
    fig.add_trace(go.Scatter3d(
        x=[carbon_values[best_biodiversity_idx]],
        y=[biodiversity_values[best_biodiversity_idx]],
        z=[cost_values[best_biodiversity_idx]],
        mode='markers',
        marker=dict(
            size=10,
            color='green',
            symbol='diamond'
        ),
        name='Best Biodiversity'
    ))
    
    # Add best cost solution
    fig.add_trace(go.Scatter3d(
        x=[carbon_values[best_cost_idx]],
        y=[biodiversity_values[best_cost_idx]],
        z=[cost_values[best_cost_idx]],
        mode='markers',
        marker=dict(
            size=10,
            color='orange',
            symbol='diamond'
        ),
        name='Lowest Cost'
    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Carbon Sequestration',
            yaxis_title='Biodiversity Index',
            zaxis_title='Implementation Cost',
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_tree_distribution(solution, title):
    """Plot tree type distribution as a pie chart"""
    tree_types = solution['tree_types'][0]
    labels = [f'Type {i+1}' for i in range(len(tree_types))]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=tree_types,
        hole=.3,
        marker_colors=px.colors.sequential.Greens
    )])
    
    fig.update_layout(
        title=title,
    )
    
    return fig

def plot_optimization_history(history):
    """Plot optimization progress over generations"""
    generations = [h['generation'] for h in history]
    avg_carbon = [h['avg_carbon'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    pareto_sizes = [h['pareto_size'] for h in history]
    
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Optimization Progress", "Pareto Front Size"))
    
    # Top plot
    fig.add_trace(
        go.Scatter(x=generations, y=avg_carbon, name="Avg Carbon", line=dict(color="green")),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=generations, y=best_fitness, name="Best Fitness", line=dict(color="blue")),
        row=1, col=1
    )
    
    # Bottom plot
    fig.add_trace(
        go.Scatter(x=generations, y=pareto_sizes, name="Pareto Front Size", line=dict(color="orange")),
        row=2, col=1
    )
    
    fig.update_layout(height=600, width=800)
    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Size", row=2, col=1)
    
    return fig

def compare_solutions(solutions):
    """Create a comparison bar chart for top solutions"""
    # Find the best solutions
    carbon_solutions = sorted(solutions, key=lambda x: x['carbon_sequestration'], reverse=True)
    biodiversity_solutions = sorted(solutions, key=lambda x: x['biodiversity_index'], reverse=True)
    cost_solutions = sorted(solutions, key=lambda x: x['implementation_cost'])
    
    best_carbon = carbon_solutions[0]
    best_biodiversity = biodiversity_solutions[0]
    best_cost = cost_solutions[0]
    
    solution_names = ['Best Carbon', 'Best Biodiversity', 'Lowest Cost']
    solutions_to_plot = [best_carbon, best_biodiversity, best_cost]
    
    # Create comparison figure
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=("Carbon Sequestration (tons/ha)", 
                                        "Biodiversity Index (0-10)", 
                                        "Implementation Cost ($)"))
    
    carbon_values = [s['carbon_sequestration'] for s in solutions_to_plot]
    fig.add_trace(
        go.Bar(x=solution_names, y=carbon_values, marker_color='green'),
        row=1, col=1
    )
    
    biodiversity_values = [s['biodiversity_index'] for s in solutions_to_plot]
    fig.add_trace(
        go.Bar(x=solution_names, y=biodiversity_values, marker_color='blue'),
        row=1, col=2
    )
    
    cost_values = [s['implementation_cost'] for s in solutions_to_plot]
    fig.add_trace(
        go.Bar(x=solution_names, y=cost_values, marker_color='orange'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig, best_carbon, best_biodiversity, best_cost

def main():
    st.markdown("<div class='main-header'>🌳 Advanced Afforestation Planning System</div>", unsafe_allow_html=True)
    
    st.markdown("""
    This application uses Generative Adversarial Networks (GANs) and Genetic Algorithms to optimize afforestation plans.
    The system balances multiple objectives including carbon sequestration, biodiversity, and implementation cost.
    """)
    
    # Sidebar options
    st.sidebar.title("Settings")
    
    # Step 1: Initialize Data
    st.sidebar.markdown("### Step 1: Data")
    data_option = st.sidebar.radio(
        "Select data source",
        ["Generate synthetic data", "Upload custom data (not implemented)"]
    )
    
    if data_option == "Generate synthetic data":
        num_samples = st.sidebar.slider("Number of samples", 100, 1000, 500)
        data = generate_synthetic_data(num_samples)
        
        with st.expander("View synthetic data"):
            st.dataframe(data.head(10))
            st.write(f"Total samples: {len(data)}")
            
    else:
        st.sidebar.warning("Custom data upload is not implemented in this demo.")
        data = generate_synthetic_data(500)
    
    # Define feature columns
    feature_columns = [
        'elevation', 'slope', 'precipitation',
        'temperature', 'soil_quality', 'current_vegetation'
    ]
    
    # Step 2: Model Training
    st.sidebar.markdown("### Step 2: Model")
    
    if st.sidebar.button("Load/Train Model"):
        st.markdown("<div class='sub-header'>Model Training</div>", unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        with st.spinner("Training model (this may take a while)..."):
            progress_text.text("Initializing model...")
            system = load_or_train_model(data, feature_columns, progress_bar)
            st.session_state['system'] = system
            progress_text.text("Model ready!")
        
    # Step 3: Generate Plans
    st.sidebar.markdown("### Step 3: Generate Plans")
    
    # Environmental constraints
    st.sidebar.markdown("Environmental Constraints:")
    max_density = st.sidebar.slider("Max tree density (trees/ha)", 50, 150, 80)
    min_biodiversity = st.sidebar.slider("Min biodiversity index", 1.0, 8.0, 3.0)
    max_cost = st.sidebar.slider("Max implementation cost ($)", 2000, 10000, 5000)
    
    environmental_constraints = {
        'max_density': max_density,
        'min_biodiversity': min_biodiversity,
        'max_cost': max_cost
    }
    
    if st.sidebar.button("Generate Afforestation Plans"):
        if 'system' not in st.session_state:
            st.error("Please load/train the model first!")
        else:
            with st.spinner("Generating optimized afforestation plans..."):
                system = st.session_state['system']
                solutions, history = system.generate_afforestation_plan(
                    environmental_constraints=environmental_constraints
                )
                st.session_state['solutions'] = solutions
                st.session_state['history'] = history
                st.success(f"Generated {len(solutions)} potential afforestation plans!")
    
    # Results visualization
    if 'solutions' in st.session_state and 'history' in st.session_state:
        solutions = st.session_state['solutions']
        history = st.session_state['history']
        
        st.markdown("<div class='sub-header'>Optimization Results</div>", unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Pareto Front", "Tree Distributions", "Optimization Progress"])
        
        with tab1:
            # Summary metrics
            fig, best_carbon, best_biodiversity, best_cost = compare_solutions(solutions)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Best Carbon Solution")
                st.markdown(f"<div class='metric-value'>{best_carbon['carbon_sequestration']:.2f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Carbon Sequestration (tons/ha)</div>", unsafe_allow_html=True)
                st.markdown(f"Biodiversity: {best_carbon['biodiversity_index']:.2f}/10")
                st.markdown(f"Cost: ${best_carbon['implementation_cost']:.2f}")
                st.markdown(f"Density: {best_carbon['density']:.2f} trees/ha")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Best Biodiversity Solution")
                st.markdown(f"<div class='metric-value'>{best_biodiversity['biodiversity_index']:.2f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Biodiversity Index (0-10)</div>", unsafe_allow_html=True)
                st.markdown(f"Carbon: {best_biodiversity['carbon_sequestration']:.2f} tons/ha")
                st.markdown(f"Cost: ${best_biodiversity['implementation_cost']:.2f}")
                st.markdown(f"Density: {best_biodiversity['density']:.2f} trees/ha")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Lowest Cost Solution")
                st.markdown(f"<div class='metric-value'>${best_cost['implementation_cost']:.2f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Implementation Cost</div>", unsafe_allow_html=True)
                st.markdown(f"Carbon: {best_cost['carbon_sequestration']:.2f} tons/ha")
                st.markdown(f"Biodiversity: {best_cost['biodiversity_index']:.2f}/10")
                st.markdown(f"Density: {best_cost['density']:.2f} trees/ha")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            # 3D Pareto front visualization
            st.markdown("### Pareto Front Visualization")
            pareto_fig = plot_3d_pareto_front(solutions)
            st.plotly_chart(pareto_fig, use_container_width=True)
            st.markdown("""
            This 3D visualization shows the Pareto front of solutions. Each point represents an afforestation plan with 
            its performance across three objectives: carbon sequestration, biodiversity, and cost. The highlighted points 
            show the best solutions for each individual objective.
            """)
        
        with tab3:
            # Tree distribution charts
            st.markdown("### Tree Type Distributions")
            st.markdown("These charts show the proportion of different tree types in each solution.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tree_dist_carbon = plot_tree_distribution(best_carbon, "Best Carbon Solution")
                st.plotly_chart(tree_dist_carbon, use_container_width=True)
            
            with col2:
                tree_dist_biodiv = plot_tree_distribution(best_biodiversity, "Best Biodiversity Solution")
                st.plotly_chart(tree_dist_biodiv, use_container_width=True)
            
            with col3:
                tree_dist_cost = plot_tree_distribution(best_cost, "Lowest Cost Solution")
                st.plotly_chart(tree_dist_cost, use_container_width=True)
        
        with tab4:
            # Optimization progress
            st.markdown("### Optimization Progress")
            history_fig = plot_optimization_history(history)
            st.plotly_chart(history_fig, use_container_width=True)
            st.markdown("""
            These charts show how the optimization algorithm progressed over generations. The top chart shows the 
            average carbon sequestration and best fitness values, while the bottom chart shows how the size of the 
            Pareto front evolved over time.
            """)
    
    # Add information about the model
    with st.expander("About this system"):
        st.markdown("""
        ### Advanced Afforestation Planning System
        
        This system uses a combination of machine learning techniques to optimize afforestation plans:
        
        1. **Generative Adversarial Networks (GANs)**: Used to learn the complex relationships between environmental features and afforestation outcomes.
        
        2. **Genetic Algorithms for Multi-Objective Optimization**: Used to find optimal trade-offs between carbon sequestration, biodiversity, and implementation cost.
        
        3. **Pareto Front Analysis**: Identifies solutions that represent optimal trade-offs where improving one objective would necessarily worsen another.
        
        The model considers various environmental factors such as elevation, slope, precipitation, temperature, soil quality, 
        and current vegetation to recommend the best tree planting strategies.
        """)

if __name__ == "__main__":
    main()