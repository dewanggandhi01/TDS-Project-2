from bs4 import BeautifulSoup
import requests
import re
import sys
import io
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import base64
from typing import Dict, Any, Optional


def scrape_html(tag: str, url: str) -> str:
    """
    Scrape the HTML content of a given tag from a URL.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    elements = soup.find_all(tag)
    
    return ' '.join([element.get_text() for element in elements])


def scrape_table_html(url: str, max_tokens: int = 30000) -> str:
    """
    Scrape relevant HTML tables from a URL, optimized for LLM/ReAct agent usage.
    Only tables with meaningful data (at least 2 rows and 2 columns) are included.
    Token budget is respected to avoid overflows.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all("table")
    table_strings = []
    total_tokens = 0

    def count_tokens(text):
        # Approximate token count: 1 token ≈ 4 chars (for English text)
        return max(1, len(text) // 4)

    for table in tables:
        # Filter out tables that are too small to be useful
        rows = table.find_all('tr')
        if len(rows) < 2:
            continue
        first_row_cells = rows[0].find_all(['th', 'td'])
        if len(first_row_cells) < 2:
            continue

        row_strings = []
        for row in rows:
            cells = row.find_all(['th', 'td'])
            cell_text = []
            for cell in cells:
                # Remove footnotes, links, and superscripts
                for tag in cell.find_all(['sup', 'a']):
                    tag.decompose()
                text = cell.get_text(separator=' ', strip=True)
                text = re.sub(r'\[\w+\]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    cell_text.append(text)
            if cell_text:
                row_text = ' | '.join(cell_text)
                row_tokens = count_tokens(row_text)
                if total_tokens + row_tokens > max_tokens:
                    break
                row_strings.append(row_text)
                total_tokens += row_tokens
        if row_strings:
            table_text = '\n'.join(row_strings)
            table_tokens = count_tokens(table_text)
            if total_tokens + table_tokens > max_tokens:
                break
            table_strings.append(table_text)
            total_tokens += table_tokens
        if total_tokens >= max_tokens:
            break

    if not table_strings:
        return "No relevant tables found or token budget exceeded."
    return '\n\n---\n\n'.join(table_strings)


def read_csv(file_name: str) -> str:
    """
    Read a CSV file and return its content as a string.
    """
    try:
        with open(os.getcwd()+"/"+file_name, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"File {file_name} not found."
    except Exception as e:
        return f"An error occurred while reading the file: {str(e)}"


class NetworkAnalyzer:
    def __init__(self, edges_file: str):
        """Initialize network analyzer with edges CSV file."""
        self.edges_df = pd.read_csv(edges_file)
        self.G = self._create_graph()
    
    def _create_graph(self) -> nx.Graph:
        """Create undirected graph from edges DataFrame."""
        G = nx.Graph()
        
        # Assuming edges.csv has columns: source, target
        if 'source' in self.edges_df.columns and 'target' in self.edges_df.columns:
            edges = list(zip(self.edges_df['source'], self.edges_df['target']))
        else:
            # Handle case where columns might be named differently
            cols = self.edges_df.columns
            if len(cols) >= 2:
                edges = list(zip(self.edges_df.iloc[:, 0], self.edges_df.iloc[:, 1]))
            else:
                raise ValueError("CSV must have at least 2 columns for source and target")
        
        G.add_edges_from(edges)
        return G
    
    def get_edge_count(self) -> int:
        """Get the number of edges in the network."""
        return self.G.number_of_edges()
    
    def get_highest_degree_node(self) -> str:
        """Get the node with the highest degree."""
        if not self.G.nodes():
            return None
        degrees = dict(self.G.degree())
        return max(degrees, key=degrees.get)
    
    def get_average_degree(self) -> float:
        """Get the average degree of the network."""
        if not self.G.nodes():
            return 0.0
        degrees = [d for n, d in self.G.degree()]
        return sum(degrees) / len(degrees)
    
    def get_density(self) -> float:
        """Get the network density."""
        return nx.density(self.G)
    
    def get_shortest_path_length(self, source: str, target: str) -> Optional[int]:
        """Get the shortest path length between two nodes."""
        try:
            return nx.shortest_path_length(self.G, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def create_network_plot(self) -> str:
        """Create a network visualization and return as base64 PNG."""
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.G, k=1, iterations=50)
        
        # Draw the network
        nx.draw_networkx_nodes(self.G, pos, node_color='lightblue', 
                              node_size=500, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(self.G, pos, font_size=10)
        
        plt.title("Network Graph")
        plt.axis('off')
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def create_degree_histogram(self) -> str:
        """Create a degree distribution histogram and return as base64 PNG."""
        plt.figure(figsize=(10, 6))
        
        degrees = [d for n, d in self.G.degree()]
        
        # Create histogram with green bars
        plt.hist(degrees, bins=range(max(degrees) + 2), color='green', 
                alpha=0.7, edgecolor='black')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def analyze_network(self) -> Dict[str, Any]:
