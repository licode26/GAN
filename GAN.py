#advanced afforestation planning system using generative adversarial networks (GANs) and genetic algorithms for multi-objective optimization.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class EnhancedAfforestationDataset(Dataset):
    """Enhanced Dataset for Afforestation Planning"""
    def __init__(self, data, feature_columns, target_columns=None, transform=None):
        self.transform = transform
        self.target_columns = ['carbon_sequestration'] if target_columns is None else target_columns

        self.feature_scaler = StandardScaler()
        features = self.feature_scaler.fit_transform(data[feature_columns])

        self.target_scaler = MinMaxScaler()
        targets = self.target_scaler.fit_transform(data[self.target_columns])

        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

        self.feature_names = feature_columns

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.features[idx]), self.targets[idx]
        return self.features[idx], self.targets[idx]

    def inverse_transform_features(self, features):

        return self.feature_scaler.inverse_transform(features.cpu().detach().numpy())

    def inverse_transform_targets(self, targets):

        return self.target_scaler.inverse_transform(targets.cpu().detach().numpy())

class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features)
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  
        return self.activation(out)

class EnhancedGenerator(nn.Module):

    def __init__(self, noise_dim, feature_dim, output_dim, soil_types=10, climate_zones=8):
        super().__init__()


        self.soil_embedding = nn.Embedding(soil_types, 16)
        self.climate_embedding = nn.Embedding(climate_zones, 16)

        self.noise_processor = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            ResidualBlock(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256)
        )


        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)


        self.feature_generator = nn.Sequential(
            ResidualBlock(256),
            nn.Linear(256, feature_dim),
            nn.Tanh()
        )


        self.tree_type_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1)  
        )

        self.plantation_density_predictor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  
        )

        self.carbon_sequestration_estimator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Softplus() 
        )

        self.biodiversity_estimator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

        self.cost_estimator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Softplus()  
        )

    def forward(self, noise, soil_type=None, climate_zone=None):
        batch_size = noise.size(0)


        features = self.noise_processor(noise)


        if soil_type is not None:
            soil_embedding = self.soil_embedding(soil_type)
            features = features + soil_embedding.unsqueeze(1)

        if climate_zone is not None:
            climate_embedding = self.climate_embedding(climate_zone)
            features = features + climate_embedding.unsqueeze(1)


        features_unsqueezed = features.unsqueeze(0)
        attn_output, _ = self.attention(features_unsqueezed, features_unsqueezed, features_unsqueezed)
        features = features + attn_output.squeeze(0)


        final_features = self.feature_generator(features)


        tree_types = self.tree_type_predictor(final_features)
        plantation_density = self.plantation_density_predictor(final_features) * 100  
        carbon_sequestration = self.carbon_sequestration_estimator(final_features) * 20  
        biodiversity_index = self.biodiversity_estimator(final_features) * 10
        implementation_cost = self.cost_estimator(final_features) * 1000 

        return (
            final_features,
            tree_types,
            plantation_density,
            carbon_sequestration,
            biodiversity_index,
            implementation_cost
        )

class EnhancedDiscriminator(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feature_dim, 512)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            ResidualBlock(256),

            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.utils.spectral_norm(nn.Linear(128, 1)),
            nn.Sigmoid()
        )

    def forward(self, features):
        validity = self.model(features)
        return validity

class AdvancedGeneticOptimizer:

    def __init__(self, population_size=200, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.history = [] 

    def initialize_population(self, generator, noise_dim, device):

        population = []
        for _ in range(self.population_size):

            noise = torch.randn(1, noise_dim).to(device)

            with torch.no_grad():
                features, tree_types, density, carbon, biodiversity, cost = generator(noise)

                population.append({
                    'noise': noise,
                    'features': features,
                    'tree_types': tree_types,
                    'density': density,
                    'carbon': carbon,
                    'biodiversity': biodiversity,
                    'cost': cost
                })

        return population

    def pareto_dominance(self, ind1, ind2):
        """Check if individual 1 dominates individual 2 (better in at least one objective, not worse in others)"""
        better_in_one = False


        if ind1['carbon'].item() > ind2['carbon'].item():
            better_in_one = True
        elif ind1['carbon'].item() < ind2['carbon'].item():
            return False

        if ind1['biodiversity'].item() > ind2['biodiversity'].item():
            better_in_one = True
        elif ind1['biodiversity'].item() < ind2['biodiversity'].item():
            return False

        
        if ind1['cost'].item() < ind2['cost'].item():
            better_in_one = True
        elif ind1['cost'].item() > ind2['cost'].item():
            return False

        return better_in_one

    def calculate_pareto_front(self, population):
       
        pareto_front = []

        for ind in population:
            dominated = False

            for other_ind in population:
                if other_ind is not ind and self.pareto_dominance(other_ind, ind):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(ind)

        return pareto_front

    def calculate_crowding_distance(self, front):
     
        if len(front) <= 2:
            for ind in front:
                ind['crowding'] = float('inf')
            return

   
        for ind in front:
            ind['crowding'] = 0

        for objective in ['carbon', 'biodiversity', 'cost']:
 
            sorted_front = sorted(front, key=lambda x: x[objective].item())

            
            sorted_front[0]['crowding'] = float('inf')
            sorted_front[-1]['crowding'] = float('inf')

            
            if len(sorted_front) > 2:
                obj_range = sorted_front[-1][objective].item() - sorted_front[0][objective].item()
                if obj_range > 0:
                    for i in range(1, len(sorted_front) - 1):
                        sorted_front[i]['crowding'] += (
                            (sorted_front[i+1][objective].item() - sorted_front[i-1][objective].item()) / obj_range
                        )

    def tournament_selection(self, population):

        selected = []

        while len(selected) < self.population_size // 2:
            candidates = random.sample(population, 2)

  
            non_dominated_candidates = self.calculate_pareto_front(candidates)

            if len(non_dominated_candidates) == 1:
                winner = non_dominated_candidates[0]
            else:
        
                winner = max(non_dominated_candidates, key=lambda x: x.get('crowding', 0))

            selected.append(winner)

        return selected

    def crossover(self, parent1, parent2):

        eta = 1

     
        noise1 = parent1['noise']
        noise2 = parent2['noise']

    
        u = torch.rand_like(noise1)

        beta = torch.ones_like(u)
        beta[u <= 0.5] = (2 * u[u <= 0.5]) ** (1 / (eta + 1))
        beta[u > 0.5] = (1 / (2 * (1 - u[u > 0.5]))) ** (1 / (eta + 1))


        child1_noise = 0.5 * ((1 + beta) * noise1 + (1 - beta) * noise2)
        child2_noise = 0.5 * ((1 - beta) * noise1 + (1 + beta) * noise2)

        return child1_noise, child2_noise

    def mutate(self, noise, mutation_strength=0.1):

        if random.random() < self.mutation_rate:
    
            mutation = torch.randn_like(noise) * mutation_strength

            mutated_noise = noise + mutation

            return mutated_noise

        return noise

    def optimize(self, generator, environmental_constraints, device):

        population = self.initialize_population(generator, noise_dim=64, device=device)


        best_solution = None
        best_fitness = float('-inf')

        for generation in range(self.generations):
 
            pareto_front = self.calculate_pareto_front(population)

            self.calculate_crowding_distance(pareto_front)

            for ind in population:
    
                if ind['density'].item() > environmental_constraints.get('max_density', 100):
                    ind['constraint_violation'] = ind['density'].item() - environmental_constraints.get('max_density', 100)
                else:
                    ind['constraint_violation'] = 0

    
                ind['fitness'] = (
                    ind['carbon'].item() * 0.5 +
                    ind['biodiversity'].item() * 0.3 -
                    ind['cost'].item() * 0.01 -
                    ind['constraint_violation'] * 10
                )

     
                if ind['fitness'] > best_fitness and ind['constraint_violation'] == 0:
                    best_fitness = ind['fitness']
                    best_solution = ind

            avg_carbon = sum(ind['carbon'].item() for ind in population) / len(population)
            self.history.append({
                'generation': generation,
                'avg_carbon': avg_carbon,
                'best_fitness': best_fitness,
                'pareto_size': len(pareto_front)
            })

         
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.2f}, "
                      f"Avg Carbon = {avg_carbon:.2f}, "
                      f"Pareto front size = {len(pareto_front)}")

     
            selected = self.tournament_selection(population)

            new_population = []

     
            elite_size = min(len(pareto_front), self.population_size // 10)
            elite = sorted(pareto_front, key=lambda x: x.get('fitness', float('-inf')), reverse=True)[:elite_size]
            new_population.extend(elite)

           
            while len(new_population) < self.population_size:
              
                parent1, parent2 = random.sample(selected, 2)

                
                child1_noise, child2_noise = self.crossover(parent1, parent2)

                
                child1_noise = self.mutate(child1_noise)
                child2_noise = self.mutate(child2_noise)

   
                for child_noise in [child1_noise, child2_noise]:
                    if len(new_population) < self.population_size:
                        with torch.no_grad():
                            features, tree_types, density, carbon, biodiversity, cost = generator(child_noise)

                            new_population.append({
                                'noise': child_noise,
                                'features': features,
                                'tree_types': tree_types,
                                'density': density,
                                'carbon': carbon,
                                'biodiversity': biodiversity,
                                'cost': cost
                            })


            population = new_population

        final_pareto = self.calculate_pareto_front(population)
        return final_pareto, self.history

class AdvancedAfforestationPlanningSystem:
    """Advanced Afforestation Planning Framework with Multi-Objective Optimization"""
    def __init__(self, feature_dim=5, noise_dim=64, tree_types=5):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")


        self.generator = EnhancedGenerator(
            noise_dim=noise_dim,
            feature_dim=feature_dim,
            output_dim=tree_types
        ).to(self.device)

        self.discriminator = EnhancedDiscriminator(feature_dim).to(self.device)


        self.g_optimizer = optim.AdamW(
            self.generator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999),
            weight_decay=1e-5
        )

        self.d_optimizer = optim.AdamW(
            self.discriminator.parameters(),
            lr=0.0001,  
            betas=(0.5, 0.999),
            weight_decay=1e-5
        )

 
        self.g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        self.d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

 
        self.adversarial_loss = nn.BCELoss()
        self.regression_loss = nn.MSELoss()
        self.wasserstein_distance = lambda real, fake: torch.mean(real) - torch.mean(fake)

   
        self.genetic_optimizer = AdvancedGeneticOptimizer(
            population_size=200,
            generations=100,
            mutation_rate=0.1
        )

  
        self.training_history = {
            'd_loss': [],
            'g_loss': [],
            'carbon_loss': [],
            'epochs': []
        }

       
        self.dataset = None

    def train(self, data, feature_columns, epochs=300, batch_size=64, validation_split=0.1):
        """Enhanced training with validation, early stopping, and regularization"""
  
        dataset = EnhancedAfforestationDataset(data, feature_columns)
        self.dataset = dataset  

        # Split into train/validation
        dataset_size = len(dataset)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
           
            self.generator.train()
            self.discriminator.train()

            epoch_d_loss = 0
            epoch_g_loss = 0
            epoch_carbon_loss = 0
            batches = 0

            for real_features, real_targets in train_loader:
                batch_size = real_features.size(0)
                batches += 1

            
                real_features = real_features.to(self.device)
                real_targets = real_targets.to(self.device)

              
                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)

           
                valid = valid - 0.1 * torch.rand_like(valid)
                fake = fake + 0.1 * torch.rand_like(fake)

            
                noise = torch.randn(batch_size, 64).to(self.device)

               
                self.d_optimizer.zero_grad()

           
                gen_features, _, _, gen_carbon, _, _ = self.generator(noise)

               
                real_validity = self.discriminator(real_features)
                d_real_loss = self.adversarial_loss(real_validity, valid)

              
                fake_validity = self.discriminator(gen_features.detach())
                d_fake_loss = self.adversarial_loss(fake_validity, fake)

               
                d_loss = (d_real_loss + d_fake_loss) / 2

            
                lambda_gp = 10
                epsilon = torch.rand(batch_size, 1).to(self.device)
                interpolated = epsilon * real_features + (1 - epsilon) * gen_features.detach()
                interpolated.requires_grad_(True)

                interpolated_validity = self.discriminator(interpolated)

                gradients = torch.autograd.grad(
                    outputs=interpolated_validity,
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(interpolated_validity).to(self.device),
                    create_graph=True,
                    retain_graph=True,
                )[0]

                gradients = gradients.view(batch_size, -1)
                gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

           
                d_loss = d_loss + gradient_penalty

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.d_optimizer.step()

               
                for _ in range(2): 
                    self.g_optimizer.zero_grad()

                  
                    noise = torch.randn(batch_size, 64).to(self.device)
                    gen_features, _, _, gen_carbon, _, _ = self.generator(noise)

                   
                    fake_validity = self.discriminator(gen_features)
                    g_adv_loss = self.adversarial_loss(fake_validity, valid)

                 
                    carbon_loss = self.regression_loss(gen_carbon, real_targets)

                  
                    feature_loss = self.regression_loss(
                        gen_features.mean(0),
                        real_features.mean(0)
                    )

                  
                    g_loss = (
                        g_adv_loss * 1.0 +
                        carbon_loss * 2.0 +
                        feature_loss * 0.1
                    )

                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    self.g_optimizer.step()

                    epoch_g_loss += g_loss.item()
                    epoch_carbon_loss += carbon_loss.item()

                epoch_d_loss += d_loss.item()

       
            epoch_d_loss /= batches
            epoch_g_loss /= (batches * 2)  
            epoch_carbon_loss /= (batches * 2)

        
            self.generator.eval()
            self.discriminator.eval()

            val_g_loss = 0
            val_batches = 0

            with torch.no_grad():
                for real_features, real_targets in val_loader:
                    batch_size = real_features.size(0)
                    val_batches += 1

               
                    real_features = real_features.to(self.device)
                    real_targets = real_targets.to(self.device)
                    valid = torch.ones(batch_size, 1).to(self.device)

               
                    noise = torch.randn(batch_size, 64).to(self.device)
                    gen_features, _, _, gen_carbon, _, _ = self.generator(noise)

                    fake_validity = self.discriminator(gen_features)
                    g_adv_loss = self.adversarial_loss(fake_validity, valid)
                    carbon_loss = self.regression_loss(gen_carbon, real_targets)
                    feature_loss = self.regression_loss(
                        gen_features.mean(0),
                        real_features.mean(0)
                    )

          
                    val_loss = (
                        g_adv_loss * 1.0 +
                        carbon_loss * 2.0 +
                        feature_loss * 0.1
                    )

                    val_g_loss += val_loss.item()

            val_g_loss /= val_batches

       
            self.g_scheduler.step(val_g_loss)
            self.d_scheduler.step(epoch_d_loss)

        
            self.training_history['d_loss'].append(epoch_d_loss)
            self.training_history['g_loss'].append(epoch_g_loss)
            self.training_history['carbon_loss'].append(epoch_carbon_loss)
            self.training_history['epochs'].append(epoch)

         
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"D Loss: {epoch_d_loss:.4f}, "
                      f"G Loss: {epoch_g_loss:.4f}, "
                      f"Carbon Loss: {epoch_carbon_loss:.4f}, "
                      f"Val Loss: {val_g_loss:.4f}")

              
                with torch.no_grad():
                    test_noise = torch.randn(1, 64).to(self.device)
                    _, _, test_density, test_carbon, test_biodiversity, test_cost = self.generator(test_noise)

                    print(f"Sample output - "
                          f"Carbon: {test_carbon.item():.2f}, "
                          f"Density: {test_density.item():.2f}, "
                          f"Biodiversity: {test_biodiversity.item():.2f}, "
                          f"Cost: {test_cost.item():.2f}")

         
            if val_g_loss < best_val_loss:
                best_val_loss = val_g_loss
                patience_counter = 0
               
                torch.save(self.generator.state_dict(), 'best_generator.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    
                    self.generator.load_state_dict(torch.load('best_generator.pt'))
                    break

    
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict()
        }, 'afforestation_models.pt')

    def generate_afforestation_plan(self, environmental_constraints, device=None):
        """Generate optimized afforestation plans using genetic algorithm"""
        if device is None:
            device = self.device

 
        if self.dataset is None:
            raise ValueError("Model must be trained before generating plans")


        self.generator.eval()


        pareto_front, history = self.genetic_optimizer.optimize(
            self.generator,
            environmental_constraints,
            device
    )

        solutions = []

        carbon_solutions = sorted(pareto_front, key=lambda x: x['carbon'].item(), reverse=True)[:3]
        biodiversity_solutions = sorted(pareto_front, key=lambda x: x['biodiversity'].item(), reverse=True)[:3]
        cost_solutions = sorted(pareto_front, key=lambda x: x['cost'].item())[:3]

        all_solutions = carbon_solutions + biodiversity_solutions + cost_solutions
        seen_noise = []

        for solution in all_solutions:
       
            noise_str = str(solution['noise'].cpu().numpy())
            if noise_str not in seen_noise:
                seen_noise.append(noise_str)

       
                solutions.append({
                'features': solution['features'].cpu().detach().numpy(),
                'tree_types': solution['tree_types'].cpu().detach().numpy(),
                'density': solution['density'].cpu().detach().numpy().item(),
                'carbon_sequestration': solution['carbon'].cpu().detach().numpy().item(),
                'biodiversity_index': solution['biodiversity'].cpu().detach().numpy().item(),
                'implementation_cost': solution['cost'].cpu().detach().numpy().item(),
                'fitness': solution.get('fitness', 0)
            })

        return solutions, history

def visualize_results(self, solutions, history=None):
    """Visualize optimization results and Pareto front"""
    if not solutions:
        print("No solutions to visualize")
        return


    fig = plt.figure(figsize=(18, 10))


    ax1 = fig.add_subplot(231, projection='3d')


    carbon_values = [s['carbon_sequestration'] for s in solutions]
    biodiversity_values = [s['biodiversity_index'] for s in solutions]
    cost_values = [s['implementation_cost'] for s in solutions]

    ax1.scatter(carbon_values, biodiversity_values, cost_values,
               c='blue', s=50, alpha=0.7, label='Pareto Solutions')


    best_carbon_idx = carbon_values.index(max(carbon_values))
    best_biodiversity_idx = biodiversity_values.index(max(biodiversity_values))
    best_cost_idx = cost_values.index(min(cost_values))

    ax1.scatter([carbon_values[best_carbon_idx]],
               [biodiversity_values[best_carbon_idx]],
               [cost_values[best_carbon_idx]],
               c='red', s=100, label='Best Carbon')

    ax1.scatter([carbon_values[best_biodiversity_idx]],
               [biodiversity_values[best_biodiversity_idx]],
               [cost_values[best_biodiversity_idx]],
               c='green', s=100, label='Best Biodiversity')

    ax1.scatter([carbon_values[best_cost_idx]],
               [biodiversity_values[best_cost_idx]],
               [cost_values[best_cost_idx]],
               c='orange', s=100, label='Lowest Cost')

    ax1.set_xlabel('Carbon Sequestration')
    ax1.set_ylabel('Biodiversity Index')
    ax1.set_zlabel('Implementation Cost')
    ax1.set_title('Pareto Front of Solutions')
    ax1.legend()

    ax2 = fig.add_subplot(232)
    tree_types = solutions[best_carbon_idx]['tree_types'][0]
    tree_labels = [f'Type {i+1}' for i in range(len(tree_types))]
    ax2.bar(tree_labels, tree_types)
    ax2.set_title('Tree Type Distribution (Best Carbon)')
    ax2.set_ylabel('Proportion')


    ax3 = fig.add_subplot(233)
    tree_types = solutions[best_biodiversity_idx]['tree_types'][0]
    ax3.bar(tree_labels, tree_types)
    ax3.set_title('Tree Type Distribution (Best Biodiversity)')
    ax3.set_ylabel('Proportion')

  
    ax4 = fig.add_subplot(234)
    solution_labels = ['Best Carbon', 'Best Biodiversity', 'Lowest Cost']
    carbon_values = [solutions[best_carbon_idx]['carbon_sequestration'],
                    solutions[best_biodiversity_idx]['carbon_sequestration'],
                    solutions[best_cost_idx]['carbon_sequestration']]

    biodiversity_values = [solutions[best_carbon_idx]['biodiversity_index'],
                          solutions[best_biodiversity_idx]['biodiversity_index'],
                          solutions[best_cost_idx]['biodiversity_index']]

    cost_values = [solutions[best_carbon_idx]['implementation_cost'],
                  solutions[best_biodiversity_idx]['implementation_cost'],
                  solutions[best_cost_idx]['implementation_cost']]

    x = np.arange(len(solution_labels))
    width = 0.25

    ax4.bar(x - width, carbon_values, width, label='Carbon')
    ax4.bar(x, biodiversity_values, width, label='Biodiversity')
    ax4.bar(x + width, [c/100 for c in cost_values], width, label='Cost/100')

    ax4.set_ylabel('Values')
    ax4.set_title('Comparison of Key Metrics')
    ax4.set_xticks(x)
    ax4.set_xticklabels(solution_labels)
    ax4.legend()

    if history:
        ax5 = fig.add_subplot(235)
        generations = [h['generation'] for h in history]
        avg_carbon = [h['avg_carbon'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]

        ax5.plot(generations, avg_carbon, label='Avg Carbon')
        ax5.plot(generations, best_fitness, label='Best Fitness')
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Value')
        ax5.set_title('Optimization Progress')
        ax5.legend()

        ax6 = fig.add_subplot(236)
        pareto_sizes = [h['pareto_size'] for h in history]
        ax6.plot(generations, pareto_sizes)
        ax6.set_xlabel('Generation')
        ax6.set_ylabel('Pareto Front Size')
        ax6.set_title('Pareto Front Evolution')

    plt.tight_layout()
    plt.savefig('afforestation_results.png')
    plt.show()


    print("\n===== AFFORESTATION PLANNING REPORT =====")
    print("\nBest Carbon Sequestration Solution:")
    print(f"Carbon Sequestration: {solutions[best_carbon_idx]['carbon_sequestration']:.2f} tons/ha")
    print(f"Biodiversity Index: {solutions[best_carbon_idx]['biodiversity_index']:.2f}/10")
    print(f"Implementation Cost: ${solutions[best_carbon_idx]['implementation_cost']:.2f}")
    print(f"Planting Density: {solutions[best_carbon_idx]['density']:.2f} trees/ha")

    print("\nBest Biodiversity Solution:")
    print(f"Carbon Sequestration: {solutions[best_biodiversity_idx]['carbon_sequestration']:.2f} tons/ha")
    print(f"Biodiversity Index: {solutions[best_biodiversity_idx]['biodiversity_index']:.2f}/10")
    print(f"Implementation Cost: ${solutions[best_biodiversity_idx]['implementation_cost']:.2f}")
    print(f"Planting Density: {solutions[best_biodiversity_idx]['density']:.2f} trees/ha")

    print("\nLowest Cost Solution:")
    print(f"Carbon Sequestration: {solutions[best_cost_idx]['carbon_sequestration']:.2f} tons/ha")
    print(f"Biodiversity Index: {solutions[best_cost_idx]['biodiversity_index']:.2f}/10")
    print(f"Implementation Cost: ${solutions[best_cost_idx]['implementation_cost']:.2f}")
    print(f"Planting Density: {solutions[best_cost_idx]['density']:.2f} trees/ha")

def predict_carbon_sequestration(self, features):
    """Predict carbon sequestration for specific environmental features"""
    if self.dataset is None:
        raise ValueError("Model must be trained before making predictions")


    if isinstance(features, pd.DataFrame):
      
        normalized_features = self.dataset.feature_scaler.transform(features)
        features_tensor = torch.FloatTensor(normalized_features).to(self.device)
    elif isinstance(features, np.ndarray):
        normalized_features = self.dataset.feature_scaler.transform(features)
        features_tensor = torch.FloatTensor(normalized_features).to(self.device)
    elif isinstance(features, torch.Tensor):
        features_tensor = features.to(self.device)
    else:
        raise ValueError("Features must be DataFrame, numpy array, or Tensor")


    self.generator.eval()

    with torch.no_grad():
     
        noise = torch.randn(features_tensor.size(0), 64).to(self.device)
        _, tree_types, density, carbon, biodiversity, cost = self.generator(noise)

 
        tree_types = tree_types.cpu().numpy()
        density = density.cpu().numpy()
        carbon = carbon.cpu().numpy()
        biodiversity = biodiversity.cpu().numpy()
        cost = cost.cpu().numpy()

    results = pd.DataFrame({
        'carbon_sequestration': carbon.flatten(),
        'biodiversity_index': biodiversity.flatten(),
        'implementation_cost': cost.flatten(),
        'planting_density': density.flatten()
    })


    for i in range(tree_types.shape[1]):
        results[f'tree_type_{i+1}'] = tree_types[:, i]

    return results

def load_pretrained_model(self, model_path):
    """Load a pretrained model"""
    checkpoint = torch.load(model_path, map_location=self.device)

    self.generator.load_state_dict(checkpoint['generator'])
    self.discriminator.load_state_dict(checkpoint['discriminator'])

    print(f"Loaded pretrained model from {model_path}")

    return self

def calculate_environmental_impact(self, location_data):
    """Calculate long-term environmental impact of an afforestation plan"""

    required_columns = ['soil_type', 'climate_zone', 'current_vegetation',
                        'slope', 'precipitation', 'temperature']

    for col in required_columns:
        if col not in location_data.columns:
            raise ValueError(f"Missing required column: {col}")


    predictions = self.predict_carbon_sequestration(location_data)

    carbon_30yr = predictions['carbon_sequestration'] * 30 * 0.85

    
    water_conservation = predictions['planting_density'] * 0.5 * location_data['precipitation']

  
    soil_protection = predictions['planting_density'] * 0.3 * location_data['slope']

    habitat_creation = predictions['biodiversity_index'] * 0.8

 
    predictions['carbon_30yr'] = carbon_30yr
    predictions['water_conservation'] = water_conservation.values
    predictions['soil_protection'] = soil_protection.values
    predictions['habitat_creation'] = habitat_creation

    return predictions

def save_model(self, filepath):
    """Save the current model to a file"""
    torch.save({
        'generator': self.generator.state_dict(),
        'discriminator': self.discriminator.state_dict(),
        'g_optimizer': self.g_optimizer.state_dict(),
        'd_optimizer': self.d_optimizer.state_dict(),
        'training_history': self.training_history,
    }, filepath)

    print(f"Model saved to {filepath}")

    return self

def main():
    
    print("Generating synthetic afforestation data...")
    num_samples = 500

   
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

    data['carbon_sequestration'] = (
        0.2 * data['elevation'] / 1000 +
        -0.5 * data['slope'] / 45 +
        0.8 * data['precipitation'] / 2000 +
        -0.3 * (data['temperature'] - 15) / 15 +
        0.6 * data['soil_quality'] / 10 +
        0.4 * data['current_vegetation'] / 100
    ) * 15  
  
    data['carbon_sequestration'] += np.random.normal(0, 1, num_samples)
    data['carbon_sequestration'] = np.abs(data['carbon_sequestration'])  


    print("\nData Summary:")
    print(data.describe())


    feature_columns = [
        'elevation', 'slope', 'precipitation',
        'temperature', 'soil_quality', 'current_vegetation'
    ]


    print("\nInitializing Afforestation Planning System...")
    system = AdvancedAfforestationPlanningSystem(
        feature_dim=len(feature_columns),
        noise_dim=64,
        tree_types=5
    )


    print("\nTraining the system (this may take a while)...")
    try:
        system.train(
            data=data,
            feature_columns=feature_columns,
            epochs=50,  
            batch_size=32,
            validation_split=0.2
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training error: {e}")
        return


    environmental_constraints = {
        'max_density': 80, 
        'min_biodiversity': 3.0, 
        'max_cost': 5000  
    }

    print("\nGenerating optimized afforestation plans...")
    try:
        solutions, history = system.generate_afforestation_plan(
            environmental_constraints=environmental_constraints
        )
        print(f"Generated {len(solutions)} potential afforestation plans!")
    except Exception as e:
        print(f"Plan generation error: {e}")
        return


    print("\n===== TOP AFFORESTATION PLANS =====")


    carbon_solutions = sorted(solutions, key=lambda x: x['carbon_sequestration'], reverse=True)
    biodiversity_solutions = sorted(solutions, key=lambda x: x['biodiversity_index'], reverse=True)
    cost_solutions = sorted(solutions, key=lambda x: x['implementation_cost'])


    print("\nBest Carbon Sequestration Solution:")
    best_carbon = carbon_solutions[0]
    print(f"Carbon Sequestration: {best_carbon['carbon_sequestration']:.2f} tons/ha")
    print(f"Biodiversity Index: {best_carbon['biodiversity_index']:.2f}/10")
    print(f"Implementation Cost: ${best_carbon['implementation_cost']:.2f}")
    print(f"Planting Density: {best_carbon['density']:.2f} trees/ha")
    print("Tree Type Distribution:")
    for i, prop in enumerate(best_carbon['tree_types'][0]):
        print(f"  Tree Type {i+1}: {prop:.2%}")


    print("\nBest Biodiversity Solution:")
    best_biodiversity = biodiversity_solutions[0]
    print(f"Carbon Sequestration: {best_biodiversity['carbon_sequestration']:.2f} tons/ha")
    print(f"Biodiversity Index: {best_biodiversity['biodiversity_index']:.2f}/10")
    print(f"Implementation Cost: ${best_biodiversity['implementation_cost']:.2f}")
    print(f"Planting Density: {best_biodiversity['density']:.2f} trees/ha")


    print("\nLowest Cost Solution:")
    best_cost = cost_solutions[0]
    print(f"Carbon Sequestration: {best_cost['carbon_sequestration']:.2f} tons/ha")
    print(f"Biodiversity Index: {best_cost['biodiversity_index']:.2f}/10")
    print(f"Implementation Cost: ${best_cost['implementation_cost']:.2f}")
    print(f"Planting Density: {best_cost['density']:.2f} trees/ha")


    print("\nVisualizing results...")
    try:
       
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))

      
        solution_names = ['Carbon', 'Biodiversity', 'Cost']
        solutions_to_plot = [best_carbon, best_biodiversity, best_cost]

        carbon_values = [s['carbon_sequestration'] for s in solutions_to_plot]
        axs[0].bar(solution_names, carbon_values)
        axs[0].set_title('Carbon Sequestration (tons/ha)')

        biodiversity_values = [s['biodiversity_index'] for s in solutions_to_plot]
        axs[1].bar(solution_names, biodiversity_values)
        axs[1].set_title('Biodiversity Index (0-10)')

        cost_values = [s['implementation_cost'] for s in solutions_to_plot]
        axs[2].bar(solution_names, cost_values)
        axs[2].set_title('Implementation Cost ($)')

        plt.tight_layout()
        plt.savefig('afforestation_comparison.png')
        print("Visualization saved as 'afforestation_comparison.png'")

        plt.figure(figsize=(10, 6))
        tree_types = best_carbon['tree_types'][0]
        tree_labels = [f'Type {i+1}' for i in range(len(tree_types))]
        plt.bar(tree_labels, tree_types)
        plt.title('Tree Type Distribution for Best Carbon Solution')
        plt.ylabel('Proportion')
        plt.savefig('tree_type_distribution.png')
        print("Tree distribution saved as 'tree_type_distribution.png'")

    except Exception as e:
        print(f"Visualization error: {e}")

    print("\nAfforestation planning process completed!")

if __name__ == "__main__":
    main()