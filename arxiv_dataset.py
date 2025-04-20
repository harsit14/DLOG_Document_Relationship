import os
import json
from torch_geometric.data import Data, Dataset
import networkx as nx
from typing import Optional, Callable
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from datetime import datetime
import torch

class ArXivDataset(Dataset):
    def __init__(self, root: str, data_path: Optional[str] = None, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        print("Starting ArXivDataset initialization...")
        self.data_path = data_path
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Initialize text encoder
        self.title_embeddings_path = os.path.join(root, 'processed', 'title_embeddings.pt')
        self.abstract_embeddings_path = os.path.join(root, 'processed', 'abstract_embeddings.pt')
        print(f"Paths initialized: data_path={self.data_path}, title_emb={self.title_embeddings_path}, abstract_emb={self.abstract_embeddings_path}")
        
        # Initialize parent class
        print("Initializing parent Dataset class...")
        print("Parent class initialized")
        
        # Try to load existing data first
        if self.data_path is not None and os.path.exists(self.data_path):
            try:
                print(f"Attempting to load data from {self.data_path}")
                self.data = torch.load(self.data_path, weights_only=False)
                print("Data loaded successfully")
                return
            except Exception as e:
                print(f"Error loading data: {e}")
                print("Falling back to processing data from scratch...")
        
        # Process data from scratch
        print("Starting data processing from scratch...")
        self.data = self._process_data()
        print("Data processing completed")

    @property
    def raw_file_names(self):
        return ['arxiv-metadata-oai-snapshot.json']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download from Kaggle
        # Note: You'll need to set up Kaggle API credentials
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'Cornell-University/arxiv',
            path=self.raw_dir,
            unzip=True
        )

    def _process_data(self):
        print("Processing data from scratch...")
        # Read the JSON file and process papers in batches
        papers = []
        author_to_papers = {}
        paper_id_to_idx = {}
        titles = []
        abstracts = []
        paper_ids_set = set()  # Track unique paper IDs
        
        # First pass: collect valid papers and build author mapping
        with open(os.path.join(self.raw_dir, 'arxiv-metadata-oai-snapshot.json'), 'r') as f:
            for line in f:
                paper = json.loads(line)
                # Replace null values with empty strings
                paper = {k: (v if v is not None else '') for k, v in paper.items()}
                
                paper_id = paper['id']
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                # Skip papers without title or abstract
                if not title or not abstract:
                    continue
                
                # Check for duplicate paper IDs
                if paper_id in paper_ids_set:
                    print(f"Warning: Duplicate paper ID found: {paper_id}")
                    continue
                paper_ids_set.add(paper_id)
                
                # Store paper data
                papers.append(paper)
                titles.append(title)
                abstracts.append(abstract)
                paper_id_to_idx[paper_id] = len(titles) - 1
                
                # Build author mapping
                authors = paper.get('authors', '').split(', ')
                for author in authors:
                    if author not in author_to_papers:
                        author_to_papers[author] = []
                    author_to_papers[author].append(paper_id)
        
        print(f"Number of valid papers: {len(papers)}")
        print(f"Number of unique paper IDs: {len(paper_ids_set)}")
        
        # Create graph and add nodes
        G = nx.Graph()
        nodes_added = 0
        for paper in papers:
            paper_id = paper['id']
            if paper_id not in paper_ids_set:
                print(f"Warning: Paper ID {paper_id} not found in paper_ids_set")
                continue
                
            authors = paper.get('authors', '').split(', ')
            node_features = {
                'authors': authors,
                'comments': paper.get('comments', ''),
                'doi': paper.get('doi', ''),
                'journal_ref': paper.get('journal-ref', ''),
                'license': paper.get('license', ''),
                'report-no': paper.get('report-no', ''),
                'update_date': paper.get('update_date', ''),
                'versions': paper.get('versions', [])
            }
            G.add_node(paper_id, **node_features)
            nodes_added += 1
        
        print(f"Number of nodes added to graph: {nodes_added}")
        print(f"Number of nodes in graph: {G.number_of_nodes()}")
        
        # Create edges more efficiently
        for paper_ids in author_to_papers.values():
            if len(paper_ids) > 1:  # Only process authors with multiple papers
                # Create edges between consecutive papers
                for i in range(len(paper_ids) - 1):
                    G.add_edge(paper_ids[i], paper_ids[i + 1])
        
        print(f"Number of edges: {G.number_of_edges()}")
        
        # Get embeddings
        title_embeddings, abstract_embeddings = self._get_embeddings(titles, abstracts)
        
        # Prepare node features
        node_features = []
        for paper in papers:
            paper_id = paper['id']
            if paper_id not in paper_ids_set:
                continue
                
            # Get the index of this paper
            paper_idx = paper_id_to_idx[paper_id]
            
            # 1. Text-based features
            title_embedding = title_embeddings[paper_idx]
            abstract_embedding = abstract_embeddings[paper_idx]
            
            # 2. Author-based features
            authors = paper.get('authors', '').split(', ')
            num_authors = len(authors)
            author_affiliations = len([a for a in authors if '@' in a])  # Count authors with email addresses
            
            # 3. Temporal features
            update_date = paper.get('update_date', '')
            try:
                date_obj = datetime.strptime(update_date, '%Y-%m-%d')
                days_since_epoch = (date_obj - datetime(1970, 1, 1)).days
            except:
                days_since_epoch = 0
            
            # 4. Version features
            versions = paper.get('versions', [])
            num_versions = len(versions)
            
            # 5. Reference features
            doi = paper.get('doi', '')
            journal_ref = paper.get('journal-ref', '')
            has_doi = 1 if doi else 0
            has_journal_ref = 1 if journal_ref else 0
            
            # 6. License features
            license = paper.get('license', '')
            is_open_access = 1 if 'open-access' in license.lower() else 0
            
            # Combine all features
            feature_vector = np.concatenate([
                title_embedding,  # 384-dimensional embedding
                abstract_embedding,  # 384-dimensional embedding
                [num_authors],
                [author_affiliations],
                [days_since_epoch],
                [num_versions],
                [has_doi],
                [has_journal_ref],
                [is_open_access]
            ])
            
            node_features.append(feature_vector)
        
        # Convert to tensor - first convert to numpy array for efficiency
        print("Converting features to tensor...")
        node_features = np.array(node_features)
        x = torch.from_numpy(node_features).float()
        
        # Create edge index
        print("Creating edge index...")
        edges = list(G.edges())
        # Convert paper IDs to indices
        edge_index = []
        for src, dst in edges:
            src_idx = paper_id_to_idx[src]
            dst_idx = paper_id_to_idx[dst]
            edge_index.append([src_idx, dst_idx])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create PyTorch Geometric Data object
        print("Creating PyTorch Geometric Data object...")
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index)
        data.title = titles
        data.abstract = abstracts
        data.paper_id_to_idx = paper_id_to_idx

        # Process node features and create labels
        print("Processing categories...")
        categories = []
        for node_idx in range(data.num_nodes):
            paper_id = list(data.paper_id_to_idx.keys())[list(data.paper_id_to_idx.values()).index(node_idx)]
            paper = papers[node_idx]
            categories.append(paper.get('categories', ''))
        
        # Convert categories to numerical labels
        le = LabelEncoder()
        category_labels = le.fit_transform(categories)
        data.y = torch.tensor(category_labels, dtype=torch.long)
        
        # Save processed data
        torch.save(data, os.path.join(self.processed_dir, 'data.pt'))
        
        return data

    def _get_embeddings(self, titles, abstracts):
        # Check if embeddings already exist
        if os.path.exists(self.title_embeddings_path) and os.path.exists(self.abstract_embeddings_path):
            print("Loading existing embeddings...")
            title_embeddings = torch.load(self.title_embeddings_path, weights_only=False)
            abstract_embeddings = torch.load(self.abstract_embeddings_path, weights_only=False)
        else:
            # Pre-compute text embeddings for titles and abstracts
            print("Computing text embeddings...")
            title_embeddings = self.text_encoder.encode(titles, batch_size=256, show_progress_bar=True)
            print("Saving title embeddings...")
            torch.save(title_embeddings, self.title_embeddings_path, pickle_protocol=4)
            
            abstract_embeddings = self.text_encoder.encode(abstracts, batch_size=256, show_progress_bar=True)
            print("Saving abstract embeddings...")
            torch.save(abstract_embeddings, self.abstract_embeddings_path, pickle_protocol=4)
        
        return title_embeddings, abstract_embeddings

    def len(self):
        return 1  # We have one graph

    def get(self, idx):
        return self.data

    def get_paper_info(self, paper_id):
        """Get the title and abstract for a given paper ID.
        
        Args:
            paper_id (str): The ID of the paper
            
        Returns:
            tuple: (title, abstract) for the given paper ID
        """
        if paper_id not in self.data.paper_id_to_idx:
            raise ValueError(f"Paper ID {paper_id} not found in the dataset")
            
        idx = self.data.paper_id_to_idx[paper_id]
        return self.data.title[idx], self.data.abstract[idx]

def main():
    # Create dataset
    print("Creating dataset...")
    dataset = ArXivDataset(root='./data/arxiv')
    
    # Access the graph data
    data = dataset[0]
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Node features: {data.keys()}")
    print(f"Title and abstract shapes: {len(data.title)}, {len(data.abstract)}")

if __name__ == "__main__":
    main()
