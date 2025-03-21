import streamlit as st
import osmnx as ox
import networkx as nx
import numpy as np
import folium
import os
from folium.features import DivIcon
from streamlit_folium import folium_static
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.ndimage import uniform_filter1d
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union, Any

# Configuration constants
MIN_DISTANCE_KM = 5
MAX_DISTANCE_KM = 150
DISTANCE_STEP = 5
SEARCH_RADIUS_BUFFER = 1.5  # Buffer multiplier for search radius
OPENTOPODATA_URL = "https://api.opentopodata.org/v1/eudem25m?locations={locations}"
BATCH_SIZE = 400
API_PAUSE = 0.1  # Pause between API calls in seconds
ROLLING_MEAN_WINDOW = 5  # Window size for smoothing grades

class ElevationRouteOptimizer:
    """Enhanced class to handle route optimization for maximizing elevation gain with multiple strategies."""
    
    # Static class-level cache for network data
    _network_cache = {}  # Format: {(location, distance_km, activity_type): (G, metadata)}
    
    def __init__(self, activity_type: str, location: str, distance_km: float, strategy: str = "optimized"):
        """
        Initialize the route optimizer with caching support.
        
        Args:
            activity_type: Either "cycling" or "running"
            location: The starting/ending location as a string
            distance_km: Desired route distance in kilometers
            strategy: Route finding strategy - "optimized", "greedy", or "thorough" (original algorithm)
        """
        self.activity_type = activity_type
        self.location = location
        self.distance_km = distance_km
        self.network_type = "drive" if activity_type == "cycling" else "walk"
        self.strategy = strategy
        
        # Cache key for this configuration
        self.cache_key = (location, distance_km, activity_type)
        
        # Initialize with None; will be populated in fetch_network
        self.G = None
        self.route = None
        self.route_stats = None
    
    def fetch_network(self) -> None:
        """
        Fetch the street network from OSM and add elevation data with caching.
        Skip fetching if network data already exists in cache for this configuration.
        """
        # Check cache first
        if self.cache_key in self._network_cache:
            cache_entry = self._network_cache[self.cache_key]
            self.G = cache_entry['G']
            
            # Log cache hit
            st.success(f"Using cached network data for {self.location} ({len(self.G.nodes)} nodes, {len(self.G.edges)} edges)")
            
            # If using non-thorough strategy, ensure metrics are precomputed
            if self.strategy != "thorough" and not hasattr(self, 'node_elevations'):
                st.info("Precomputing graph metrics for cached network...")
                self.precompute_graph_metrics()
                
            return
            
        # Cache miss - proceed with normal network fetching
        try:
            # Determine search radius with dynamic scaling
            if self.distance_km < 10:
                buffer_factor = 1.8  # Larger buffer for short routes
            elif self.distance_km < 20:
                buffer_factor = 1.4
            elif self.distance_km < 50:
                buffer_factor = 1.2
            else:
                buffer_factor = 0.9  # Smaller buffer for longer routes
                
            search_radius = (self.distance_km * 1000 / 2) * buffer_factor
            
            # Fetch graph with multiple fallback strategies
            st.info(f"Fetching network for {self.location}...")
            
            try:
                try:
                    # Primary approach: graph from address with radius
                    point = ox.geocode(self.location)
                    self.G = ox.graph_from_point(
                        point, 
                        dist=search_radius, 
                        network_type=self.network_type,
                        simplify=True
                    )
                    st.success(f"Successfully fetched network from point with {search_radius/1000:.1f}km radius")
                except Exception as point_error:
                    # Secondary approach: direct place query
                    self.G = ox.graph_from_place(
                        self.location,
                        network_type=self.network_type,
                        simplify=True
                    )
                    st.success(f"Successfully fetched network using place name")
                    
            except Exception as place_error:
                # Final fallback: direct geocoding and bounding box
                st.warning(f"Standard fetching methods failed. Trying manual geocoding...")
                
                # Fallback to manual geocoding with Nominatim
                import geopy.geocoders
                geolocator = geopy.geocoders.Nominatim(user_agent="elevation_route_finder")
                location = geolocator.geocode(self.location)
                
                if location:
                    center_lat, center_lon = location.latitude, location.longitude
                    dist_deg = search_radius / 111000  # Approximate conversion from meters to degrees
                    north, south = center_lat + dist_deg, center_lat - dist_deg
                    east, west = center_lon + dist_deg, center_lon - dist_deg
                    
                    self.G = ox.graph_from_bbox(
                        north, south, east, west,
                        network_type=self.network_type,
                        simplify=True
                    )
                    st.success(f"Successfully fetched network using bounding box around geocoded coordinates")
                else:
                    raise ValueError(f"Could not geocode location: {self.location}")
            
            # Verify network size and connectivity
            if len(self.G.nodes) < 10:
                raise ValueError(f"Network too small ({len(self.G.nodes)} nodes). Try a different location or larger distance.")
            
            # Store original settings for elevation API
            original_elevation_url = ox.settings.elevation_url_template
            
            # Add elevation data
            try:
                # Try local elevation service first if configured
                local_elevation_url = os.environ.get(
                    "ELEVATION_API_URL", 
                    "http://localhost:5001/v1/eudem25m?locations={locations}"
                )
                
                st.info("Adding elevation data from local service...")
                try:
                    ox.settings.elevation_url_template = local_elevation_url
                    # OSMnx 2.0+ renamed the elevation functions
                    self.G = ox.elevation.add_node_elevations_google(
                        self.G, 
                        api_key=None,  # Not needed for OpenTopoData
                        batch_size=500,
                        pause=0.01
                    )
                    st.success("Successfully added elevation data from local service.")
                    local_success = True
                except Exception as e:
                    st.warning(f"Local elevation service failed: {str(e)}. Trying public API...")
                    local_success = False
                    
                # If local service failed, try the public API
                if not local_success:
                    public_api_url = "https://api.opentopodata.org/v1/eudem25m?locations={locations}"
                    ox.settings.elevation_url_template = public_api_url
                    st.info("Adding elevation data from public API...")
                    
                    self.G = ox.elevation.add_node_elevations_google(
                        self.G, 
                        api_key=None,
                        batch_size=100, 
                        pause=1.0
                    )
                    st.success("Successfully added elevation data from public API.")
                
                # Calculate edge grades from node elevations
                self.G = ox.elevation.add_edge_grades(self.G)
                
                # Validate elevation data quality
                elevations = [data.get('elevation', 0) for _, data in self.G.nodes(data=True)]
                if elevations:
                    min_elev = min(elevations)
                    max_elev = max(elevations)
                    elevation_range = max_elev - min_elev
                    
                    if elevation_range < 1.0:
                        st.warning(f"Very flat terrain detected (elevation range: {elevation_range:.1f}m). " 
                                  f"Elevation gain optimization may not be effective.")
                    else:
                        st.info(f"Terrain elevation range: {min_elev:.1f}m to {max_elev:.1f}m " 
                               f"(range: {elevation_range:.1f}m)")
            finally:
                # Restore original settings
                ox.settings.elevation_url_template = original_elevation_url
            
            # Cache the network data for future use
            self._network_cache[self.cache_key] = {
                'G': self.G,
                'timestamp': pd.Timestamp.now(),
                'node_count': len(self.G.nodes),
                'edge_count': len(self.G.edges)
            }
            
            # Precompute metrics for non-thorough strategies
            if self.strategy != "thorough":
                self.precompute_graph_metrics()
                
        except Exception as e:
            st.error(f"Error fetching network: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            raise
    def get_edge_geometry(self, u, v):
        """Return edge geometry with consistent [lat, lon] coordinate pairs."""
        edge_data = min(self.G[u][v].values(), key=lambda x: x.get('length', float('inf')))
        
        if 'geometry' in edge_data:
            # LineString from OSMnx has (lon, lat) ordering
            geom = edge_data['geometry']
            # Convert to (lat, lon) for Folium while preserving edge data
            return [(lat, lon) for lon, lat in geom.coords], edge_data
        else:
            # Explicitly maintain [lat, lon] order for Folium
            return [(self.G.nodes[u]['y'], self.G.nodes[u]['x']),
                    (self.G.nodes[v]['y'], self.G.nodes[v]['x'])], edge_data
    
    def precompute_graph_metrics(self) -> None:
        """
        Precompute common graph metrics to avoid redundant calculations during route finding.
        This significantly improves performance for larger graphs.
        """
        st.info("Precomputing graph metrics...")
        
        # 1. Precompute node elevations
        self.node_elevations = {}
        for node, data in self.G.nodes(data=True):
            self.node_elevations[node] = data.get('elevation', 0)
        
        # 2. Precompute edge grades and weights
        self.edge_grades = {}
        self.edge_lengths = {}
        self.edge_weights = {}
        
        for u, v, k, data in self.G.edges(keys=True, data=True):
            edge_id = (u, v, k)
            
            # Store grade
            grade = data.get('grade', 0)
            self.edge_grades[edge_id] = grade
            
            # Store length
            length = data.get('length', 0)
            self.edge_lengths[edge_id] = length
            
            # Compute custom weight based on grade
            if grade > 0:  # Uphill
                weight = 1.0 / (1.0 + grade*2)
            elif grade < 0:  # Downhill
                weight = 1.5 - grade/10
            else:  # Flat
                weight = 1.3
                
            # Ensure weight is reasonable
            weight = max(0.1, min(5.0, weight))
            
            # Set weight (scaled by length)
            self.edge_weights[edge_id] = weight * length
            
            # Add to graph for pathfinding
            self.G[u][v][k]['custom_weight'] = self.edge_weights[edge_id]
        
        # 3. Precompute node coordinates
        self.node_coords = {}
        for node in self.G.nodes():
            self.node_coords[node] = (self.G.nodes[node]['y'], self.G.nodes[node]['x'])
        
        # 4. Optional: Precompute distance matrix for key nodes
        # This can be expensive but speeds up subsequent distance calculations
        if len(self.G.nodes) < 1000:  # Only for reasonably sized graphs
            from geopy.distance import geodesic
            
            # Sample a subset of nodes for the distance matrix
            import random
            sample_size = min(100, len(self.G.nodes))
            self.key_nodes = random.sample(list(self.G.nodes), sample_size)
            
            # Create distance lookup
            self.node_distances = {}
            for i, node1 in enumerate(self.key_nodes):
                for node2 in self.key_nodes[i+1:]:
                    coord1 = self.node_coords[node1]
                    coord2 = self.node_coords[node2]
                    
                    dist = geodesic(coord1, coord2).meters
                    self.node_distances[(node1, node2)] = dist
                    self.node_distances[(node2, node1)] = dist
        
        # 5. Build a simplified graph representation for faster queries
        self.simplified_G = nx.Graph()

        for u, v, data in self.G.edges(data=True):
            # If edge doesn't exist yet, add it with all attributes
            if not self.simplified_G.has_edge(u, v):
                self.simplified_G.add_edge(u, v, **data)
            else:
                # If edge exists, check if the new grade is better
                existing_grade = self.simplified_G[u][v].get('grade', 0)
                new_grade = data.get('grade', 0)
                
                # If new grade is better, update the edge attributes individually
                if new_grade > existing_grade:
                    # Update attributes individually instead of replacing the entire dict
                    for key, value in data.items():
                        self.simplified_G[u][v][key] = value
        
        st.success(f"Precomputed metrics for {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
    
    def calculate_distance_between_nodes(self, node1, node2):
        """
        Faster distance calculation using precomputed values when available.
        """
        # Check if distance is precomputed
        if hasattr(self, 'node_distances') and (node1, node2) in self.node_distances:
            return self.node_distances[(node1, node2)]
        
        # Otherwise calculate it
        if hasattr(self, 'node_coords'):
            coord1 = self.node_coords[node1]
            coord2 = self.node_coords[node2]
            
            # Haversine formula is faster than geodesic for performance
            # Approximation is fine for route finding
            from math import radians, sin, cos, sqrt, asin
            
            lat1, lon1 = coord1
            lat2, lon2 = coord2
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371000  # Earth radius in meters
            
            return c * r
        else:
            # Fallback to geopy if coords not precomputed
            from geopy.distance import geodesic
            coord1 = (self.G.nodes[node1]['y'], self.G.nodes[node1]['x'])
            coord2 = (self.G.nodes[node2]['y'], self.G.nodes[node2]['x'])
            return geodesic(coord1, coord2).meters
    
    def find_optimal_route(self) -> Tuple[List, Dict[str, float]]:
        """
        Find a route that maximizes elevation gain for the given distance.
        Dispatches to appropriate algorithm based on selected strategy.
        
        Returns:
            route: List of node IDs representing the optimal route
            stats: Dictionary containing route statistics
        """
        # Select appropriate strategy
        if self.strategy == "greedy":
            return self._find_greedy_route()
        elif self.strategy == "optimized":
            return self._find_optimized_route()
        else:  # "thorough" - original algorithm
            return self._find_thorough_route()
    
    def _find_optimized_route(self) -> Tuple[List, Dict[str, float]]:
        """
        Find a route using optimized algorithm balancing performance and quality.
        
        Returns:
            route: List of node IDs representing the route
            stats: Dictionary containing route statistics
        """
        # Get node nearest to the input location as starting point
        if hasattr(self, 'location') and self.location:
            try:
                start_point = ox.geocode(self.location)
                start_node = ox.distance.nearest_nodes(
                    self.G, 
                    X=start_point[1],  # longitude
                    Y=start_point[0]   # latitude
                )
            except Exception as e:
                st.warning(f"Could not geocode starting location precisely: {str(e)}. Using graph center.")
                # Fallback: use center of the graph
                center_y = (max(node_data['y'] for _, node_data in self.G.nodes(data=True)) + 
                           min(node_data['y'] for _, node_data in self.G.nodes(data=True))) / 2
                center_x = (max(node_data['x'] for _, node_data in self.G.nodes(data=True)) + 
                           min(node_data['x'] for _, node_data in self.G.nodes(data=True))) / 2
                start_node = ox.distance.nearest_nodes(self.G, X=center_x, Y=center_y)
        else:
            # Fallback: use first node
            start_node = list(self.G.nodes)[0]
        
        # Performance optimization: Strategic target selection
        # Find elevated nodes in different directions around start point
        import numpy as np
        
        start_y, start_x = self.node_coords[start_node]
        target_distance = (self.distance_km * 1000) / 2  # Half the total distance (for out and back)
        
        # Divide the area into sectors and find best candidates in each sector
        sectors = 8  # 8 directional sectors
        sector_angle = 2 * np.pi / sectors
        sector_candidates = []
        
        # Select up to 5 highest nodes in each directional sector within distance range
        for sector in range(sectors):
            sector_center = sector * sector_angle
            sector_nodes = []
            
            for node, (y, x) in self.node_coords.items():
                if node == start_node:
                    continue
                    
                # Calculate angle and distance from start
                dx = x - start_x
                dy = y - start_y
                angle = np.arctan2(dy, dx) % (2 * np.pi)
                
                # Check if node is in this sector
                angle_diff = min(abs(angle - sector_center), 2 * np.pi - abs(angle - sector_center))
                if angle_diff <= sector_angle/2:
                    # Calculate geodesic distance
                    dist = self.calculate_distance_between_nodes(start_node, node)
                    
                    # Accept nodes within reasonable distance range
                    if 0.25 * target_distance <= dist <= 1.5 * target_distance:
                        elevation = self.node_elevations.get(node, 0)
                        sector_nodes.append((node, elevation))
            
            # Sort by elevation and take top candidates
            sector_nodes.sort(key=lambda x: x[1], reverse=True)
            sector_candidates.extend([node for node, _ in sector_nodes[:3]])  # Top 3 per sector
        
        # Add some random nodes for diversity (avoid getting trapped in local optima)
        import random
        all_nodes = list(self.G.nodes())
        if len(all_nodes) > 20:  # Only if we have enough nodes
            random_candidates = random.sample(all_nodes, min(10, len(all_nodes)))
            sector_candidates.extend(random_candidates)
        
        # Remove duplicates
        candidate_targets = list(set(sector_candidates))
        
        # Limit candidates for performance
        if len(candidate_targets) > 24:
            candidate_targets = candidate_targets[:24]
        
        # Try fewer length factors for performance
        length_factors = [0.9, 1.0, 1.1]
        
        best_route = None
        best_gain = 0
        best_stats = {}
        
        # Early termination conditions
        early_termination = False
        routes_tried = 0
        max_routes = 50  # Cap on number of routes to try
        
        st.info("Finding optimal elevation route...")
        
        for length_factor in length_factors:
            if early_termination:
                break
                
            target_length = self.distance_km * 1000 * length_factor
            
            # Shuffle candidates for more diverse exploration
            random.shuffle(candidate_targets)
            
            for target_node in candidate_targets:
                routes_tried += 1
                
                if target_node == start_node or routes_tried > max_routes:
                    continue
                    
                try:
                    # Find route out - use bidirectional Dijkstra for faster path finding
                    route_out = nx.bidirectional_dijkstra(
                        self.G, 
                        source=start_node, 
                        target=target_node, 
                        weight='custom_weight'
                    )[1]  # [1] to get the path, [0] would be the path length
                    
                    # Route validation - skip if too short
                    route_out_length = sum(self.G[u][v][0]['length'] for u, v in zip(route_out[:-1], route_out[1:]))
                    if route_out_length < target_length * 0.3:
                        continue
                    
                    # Create modified graph for return path with realistic penalties
                    temp_G = self.G.copy()
                    
                    # Less aggressive penalization for reusing edges
                    for u, v in zip(route_out[:-1], route_out[1:]):
                        edge_data = temp_G.get_edge_data(u, v)
                        for edge_key in edge_data:
                            if 'custom_weight' in edge_data[edge_key]:
                                # 3x penalty instead of 5x
                                temp_G[u][v][edge_key]['custom_weight'] *= 3.0
                    
                    # Use bidirectional Dijkstra for the return path as well
                    try:
                        route_back = nx.bidirectional_dijkstra(
                            temp_G, 
                            source=target_node, 
                            target=start_node,
                            weight='custom_weight'
                        )[1]
                    except nx.NetworkXNoPath:
                        # If bidirectional fails, try regular Dijkstra
                        route_back = nx.shortest_path(
                            temp_G, 
                            source=target_node, 
                            target=start_node, 
                            weight='custom_weight'
                        )
                    
                    # Combine routes to form a loop
                    route = route_out + route_back[1:]
                    
                    # Calculate stats
                    length, elevation_gain, elevation_loss, grades = self._calculate_route_stats(route)
                    
                    # Get elevation gain per km as a quality metric
                    gain_per_km = elevation_gain / (length/1000) if length > 0 else 0
                    
                    # Check if route is within acceptable range
                    distance_diff_pct = abs(length - self.distance_km * 1000) / (self.distance_km * 1000)
                    quality_score = gain_per_km * (1 - min(0.5, distance_diff_pct))
                    
                    # Update best route
                    current_best_gain_per_km = best_gain / (best_stats.get('length_km', 1) * 1000) if best_stats.get('length_km', 0) > 0 else 0
                    current_best_quality = current_best_gain_per_km * (1 - min(0.5, abs(best_stats.get('length_km', 0) * 1000 - self.distance_km * 1000) / (self.distance_km * 1000)))
                    
                    if distance_diff_pct <= 0.25 and quality_score > current_best_quality:
                        best_route = route
                        best_gain = elevation_gain
                        best_stats = {
                            'length_km': length / 1000,
                            'elevation_gain_m': elevation_gain,
                            'elevation_loss_m': elevation_loss,
                            'grades': grades,
                            'quality_score': quality_score
                        }
                        
                        # Early termination if we found a really good route
                        if quality_score > 20 and routes_tried > 10:
                            early_termination = True
                            break
                
                except (nx.NetworkXNoPath, KeyError) as e:
                    # Silently continue if no path exists or edge key is missing
                    continue
                except Exception as e:
                    # Log other exceptions but continue trying
                    st.warning(f"Error finding path: {str(e)}")
                    continue
        
        if best_route is None:
            st.error("Could not find a suitable route. Try adjusting parameters or choosing a different location.")
            return None, {}
        
        st.success(f"Found route with {best_stats['elevation_gain_m']:.0f}m gain after trying {routes_tried} routes")
            
        self.route = best_route
        self.route_stats = best_stats
        return best_route, best_stats
    
    def _find_greedy_route(self) -> Tuple[List, Dict[str, float]]:
        """
        Find a route using a greedy algorithm that prioritizes uphill segments.
        Implements robust edge verification to ensure graph integrity throughout path construction.
        
        Returns:
            route: List of node IDs representing the route
            stats: Dictionary containing route statistics
        """
        # Get starting node with proper error handling
        try:
            if hasattr(self, 'location') and self.location:
                start_point = ox.geocode(self.location)
                start_node = ox.distance.nearest_nodes(
                    self.G, X=start_point[1], Y=start_point[0]
                )
            else:
                # Use center of graph as fallback
                center_y = (max(node_data['y'] for _, node_data in self.G.nodes(data=True)) + 
                           min(node_data['y'] for _, node_data in self.G.nodes(data=True))) / 2
                center_x = (max(node_data['x'] for _, node_data in self.G.nodes(data=True)) + 
                           min(node_data['x'] for _, node_data in self.G.nodes(data=True))) / 2
                start_node = ox.distance.nearest_nodes(self.G, X=center_x, Y=center_y)
        except Exception as e:
            st.error(f"Error finding start node: {str(e)}")
            # Fallback to the first node in the graph
            start_node = list(self.G.nodes)[0]
        
        st.info("Finding route using greedy elevation algorithm...")
        
        # Use precomputed simplified graph if available, or create new undirected graph
        if hasattr(self, 'simplified_G') and self.simplified_G is not None:
            G_undirected = self.simplified_G
        else:
            # Create fresh undirected graph with validated edge transfers
            G_undirected = nx.Graph()
            for u, v, data in self.G.edges(data=True):
                # Safe transfer of edge data
                edge_attrs = {k: v for k, v in data.items()}
                if G_undirected.has_edge(u, v):
                    # Update attributes individually if edge exists
                    existing_grade = G_undirected[u][v].get('grade', 0)
                    new_grade = edge_attrs.get('grade', 0)
                    if new_grade > existing_grade:
                        for key, value in edge_attrs.items():
                            G_undirected[u][v][key] = value
                else:
                    # Add new edge with attributes
                    G_undirected.add_edge(u, v, **edge_attrs)
        
        # Verify start node exists in the undirected graph
        if start_node not in G_undirected:
            st.error(f"Start node {start_node} not found in the network graph.")
            # Get a valid node from the graph
            start_node = list(G_undirected.nodes)[0]
        
        # Greedy algorithm parameters
        target_distance = self.distance_km * 1000
        max_distance = target_distance * 1.2  # Allow 20% over target
        min_path_length = 0.05 * target_distance  # Minimum escape path length (5% of target)
        max_path_length = 0.2 * target_distance   # Maximum escape path length (20% of target)
        
        current_node = start_node
        route = [current_node]
        visited_edges = set()  # Track visited edges to avoid immediate backtracking
        total_distance = 0
        stuck_count = 0
        max_stuck = 5  # Max number of steps with no progress before changing strategy
        
        # Access node coordinates safely
        node_coords = {}
        if hasattr(self, 'node_coords'):
            node_coords = self.node_coords
        else:
            for node in self.G.nodes():
                try:
                    node_coords[node] = (self.G.nodes[node]['y'], self.G.nodes[node]['x'])
                except KeyError:
                    # Skip nodes missing coordinate data
                    continue
        
        # First phase: move away from start (target 40% of total distance)
        outbound_target = target_distance * 0.4
        while total_distance < outbound_target and stuck_count < max_stuck:
            # Verify current node exists in the graph
            if current_node not in G_undirected:
                st.warning(f"Node {current_node} not in graph. Reverting to previous node.")
                if len(route) > 1:
                    current_node = route[-2]  # Fall back to previous node
                else:
                    # If we can't go back, restart from a valid node
                    current_node = start_node
                continue
            
            # Get all neighbors - guaranteed to be connected due to graph structure
            neighbors = list(G_undirected.neighbors(current_node))
            
            if not neighbors:
                stuck_count += 1
                st.warning(f"Node {current_node} has no neighbors. Finding alternative path.")
                continue
            
            # Calculate edge metrics for each neighbor with robust error handling
            candidates = []
            for neighbor in neighbors:
                # Explicit verification of edge existence (defensive coding)
                if not G_undirected.has_edge(current_node, neighbor):
                    continue
                    
                if neighbor in route[-3:]:  # Avoid short cycles by not revisiting recent nodes
                    continue
                    
                try:
                    edge_data = G_undirected[current_node][neighbor]
                    grade = edge_data.get('grade', 0)
                    length = edge_data.get('length', 0)
                    
                    # Skip edges with invalid length
                    if length <= 0:
                        continue
                    
                    # Skip if edge has been visited in same direction
                    edge_key = (current_node, neighbor)
                    if edge_key in visited_edges:
                        continue
                    
                    # Simple scoring: prioritize uphill with reasonable segment length
                    if grade > 0:  # Uphill
                        score = grade * 2  # Uphill bonus
                    else:
                        score = grade * 0.5  # Downhill penalty
                        
                    # Add distance from start bonus to encourage moving away
                    start_y, start_x = node_coords.get(start_node, (0, 0))
                    neighbor_y, neighbor_x = node_coords.get(neighbor, (0, 0))
                    
                    # Safe distance calculation with fallback
                    try:
                        from geopy.distance import geodesic
                        dist_from_start = geodesic((start_y, start_x), (neighbor_y, neighbor_x)).meters
                    except Exception:
                        # Fallback to approximate Haversine formula if geodesic fails
                        from math import radians, sin, cos, sqrt, asin
                        
                        # Convert to radians
                        lat1, lon1 = radians(start_y), radians(start_x)
                        lat2, lon2 = radians(neighbor_y), radians(neighbor_x)
                        
                        # Haversine formula
                        dlon = lon2 - lon1
                        dlat = lat2 - lat1
                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        c = 2 * asin(sqrt(a))
                        r = 6371000  # Earth radius in meters
                        dist_from_start = c * r
                    
                    # Bonus for moving away from start
                    distance_bonus = min(dist_from_start / 1000, 5)  # Cap bonus at 5
                    score += distance_bonus
                    
                    candidates.append((neighbor, score, length))
                except Exception as e:
                    # Log error and skip this neighbor
                    st.warning(f"Error processing neighbor {neighbor}: {str(e)}")
                    continue
            
            if not candidates:
                stuck_count += 1
                # Handle stuck condition - look for escape paths
                escape_paths = []
                
                # Sample distant nodes to find escape paths
                try:
                    # Get a sample of potential escape nodes
                    sample_size = min(50, len(G_undirected.nodes))
                    import random
                    sample_nodes = random.sample(list(G_undirected.nodes), sample_size)
                    
                    for node in sample_nodes:
                        if node not in route[-10:]:  # Don't revisit recent nodes
                            try:
                                # Find shortest path to this node (guaranteed to use existing edges)
                                path = nx.shortest_path(G_undirected, current_node, node, weight='length')
                                
                                # Calculate path length to ensure it's reasonable
                                path_length = sum(G_undirected[u][v].get('length', 0) 
                                                 for u, v in zip(path[:-1], path[1:]))
                                
                                # Only consider paths within reasonable length range
                                if min_path_length < path_length < max_path_length:
                                    escape_paths.append((path, path_length))
                            except (nx.NetworkXNoPath, KeyError):
                                # No path or missing edge data, continue silently
                                continue
                    
                    # Sort escape paths by length (prefer shorter paths)
                    escape_paths.sort(key=lambda x: x[1])
                    
                    # Take the best escape path if any exist
                    if escape_paths:
                        escape_path, escape_length = escape_paths[0]
                        
                        # Add the path (excluding starting node which is already in route)
                        route.extend(escape_path[1:])
                        current_node = escape_path[-1]
                        total_distance += escape_length
                        st.info(f"Found escape path of length {escape_length:.1f}m")
                        stuck_count = 0
                        continue
                except Exception as e:
                    st.warning(f"Error finding escape path: {str(e)}")
                
                # If no escape path found, increment stuck counter and try again
                continue
            
            # Sort by score (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select best neighbor with fallback options
            for best_candidate in candidates:
                best_neighbor, _, edge_length = best_candidate
                
                # Verify the edge still exists (defensive)
                if G_undirected.has_edge(current_node, best_neighbor):
                    # Update route
                    route.append(best_neighbor)
                    visited_edges.add((current_node, best_neighbor))
                    visited_edges.add((best_neighbor, current_node))  # Mark both directions
                    
                    # Update total distance
                    total_distance += edge_length
                    
                    # Move to next node
                    current_node = best_neighbor
                    stuck_count = 0
                    break
            else:
                # If all candidates failed verification, increment stuck counter
                stuck_count += 1
        
        # Second phase: look for interesting terrain
        explore_target = target_distance * 0.8
        while total_distance < explore_target and stuck_count < max_stuck:
            # Verify current node exists
            if current_node not in G_undirected:
                st.warning(f"Node {current_node} not in graph. Reverting to previous node.")
                if len(route) > 1:
                    current_node = route[-2]
                    route = route[:-1]  # Remove invalid node
                else:
                    break
                continue
                
            # Now prioritize elevation gain over distance
            neighbors = list(G_undirected.neighbors(current_node))
            
            if not neighbors:
                stuck_count += 1
                continue
                    
            # Calculate edge metrics
            candidates = []
            for neighbor in neighbors:
                if neighbor in route[-3:]:  # Avoid short cycles
                    continue
                    
                # Verify edge exists
                if not G_undirected.has_edge(current_node, neighbor):
                    continue
                    
                try:
                    edge_data = G_undirected[current_node][neighbor]
                    grade = edge_data.get('grade', 0)
                    length = edge_data.get('length', 0)
                    
                    # Skip edges with invalid length
                    if length <= 0:
                        continue
                    
                    # Skip if edge has been visited recently
                    edge_key = (current_node, neighbor)
                    if edge_key in visited_edges:
                        continue
                    
                    # Scoring: heavily prioritize uphill
                    if grade > 0:
                        score = grade * 3  # Stronger uphill bonus
                    else:
                        score = grade * 0.3  # Stronger downhill penalty
                    
                    candidates.append((neighbor, score, length))
                except Exception:
                    # Skip problematic neighbors
                    continue
            
            if not candidates:
                stuck_count += 1
                # Try finding alternate paths in exploration phase too
                try:
                    # Sample nodes to find diverse paths
                    sample_size = min(30, len(G_undirected.nodes))
                    import random
                    sample_nodes = random.sample(list(G_undirected.nodes), sample_size)
                    
                    for node in sample_nodes:
                        if node not in route[-10:]:
                            try:
                                path = nx.shortest_path(G_undirected, current_node, node, weight='length')
                                if len(path) >= 3:
                                    path_length = sum(G_undirected[u][v].get('length', 0) 
                                                     for u, v in zip(path[:-1], path[1:]))
                                    if path_length < max_path_length:
                                        route.extend(path[1:])
                                        current_node = path[-1]
                                        total_distance += path_length
                                        stuck_count = 0
                                        break
                            except (nx.NetworkXNoPath, KeyError):
                                continue
                except Exception as e:
                    st.warning(f"Error in exploration phase: {str(e)}")
                continue
            
            # Sort by score
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select best neighbor with verification
            for candidate in candidates:
                neighbor, _, length = candidate
                if G_undirected.has_edge(current_node, neighbor):
                    # Update route
                    route.append(neighbor)
                    visited_edges.add((current_node, neighbor))
                    visited_edges.add((neighbor, current_node))
                    
                    # Update distance
                    total_distance += length
                    
                    # Move to next node
                    current_node = neighbor
                    stuck_count = 0
                    break
            else:
                stuck_count += 1
        
        # Final phase: return to start with robust verification
        if total_distance < max_distance:
            try:
                # Verify that current_node and start_node exist
                if current_node not in G_undirected or start_node not in G_undirected:
                    st.warning("Invalid nodes detected when returning to start.")
                else:
                    # Find path back to start (this uses existing edges)
                    return_path = nx.shortest_path(G_undirected, current_node, start_node, weight='length')
                    
                    # Calculate return path length safely
                    return_length = 0
                    for i in range(len(return_path) - 1):
                        u, v = return_path[i], return_path[i+1]
                        if G_undirected.has_edge(u, v):
                            return_length += G_undirected[u][v].get('length', 0)
                    
                    if total_distance + return_length <= max_distance:
                        route.extend(return_path[1:])  # Add all but first node (already in route)
                        total_distance += return_length
                    else:
                        # Need a shorter return path - fallback to direct length-based path
                        try:
                            # Create temporary weights focusing exclusively on path length
                            temp_G = nx.Graph()
                            for u, v, data in self.G.edges(data=True):
                                length = data.get('length', 0)
                                if length > 0:
                                    temp_G.add_edge(u, v, weight=length)
                            
                            # Find shortest path in the temporary graph
                            if current_node in temp_G and start_node in temp_G:
                                alt_return_path = nx.shortest_path(temp_G, current_node, start_node, weight='weight')
                                
                                # Verify integrity of the return path and add to route
                                valid_path = []
                                for u, v in zip(alt_return_path[:-1], alt_return_path[1:]):
                                    if G_undirected.has_edge(u, v):
                                        if not valid_path:
                                            valid_path.append(u)
                                        valid_path.append(v)
                                
                                if valid_path and valid_path[0] == current_node and valid_path[-1] == start_node:
                                    route.extend(valid_path[1:])  # Skip the first node which is current_node
                                    # Recalculate actual length
                                    return_length = sum(G_undirected[u][v].get('length', 0) 
                                                       for u, v in zip(valid_path[:-1], valid_path[1:]))
                                    total_distance += return_length
                                    st.info(f"Found alternative return path of length {return_length:.1f}m")
                        except Exception as e:
                            st.warning(f"Error finding alternative return path: {str(e)}")
            except (nx.NetworkXNoPath, KeyError, ValueError) as e:
                st.warning(f"Could not find path back to start: {str(e)}. Route may not be a complete loop.")
        
        # Verify route integrity before calculating statistics
        verified_route = []
        for i, node in enumerate(route):
            # Include the first node always
            if i == 0:
                verified_route.append(node)
                continue
                
            # For subsequent nodes, verify edge existence with previous node
            prev_node = verified_route[-1]
            if G_undirected.has_edge(prev_node, node):
                verified_route.append(node)
            else:
                # Try to find a connecting path
                try:
                    connecting_path = nx.shortest_path(G_undirected, prev_node, node, weight='length')
                    # Add intermediate nodes if any (skip the first which is prev_node)
                    if len(connecting_path) > 2:
                        verified_route.extend(connecting_path[1:])
                    else:
                        # Log disconnected segment and continue
                        st.warning(f"Disconnected nodes {prev_node}, {node} with no connecting path")
                except nx.NetworkXNoPath:
                    st.warning(f"Could not connect nodes {prev_node}, {node} - route will have gaps")
        
        # Calculate stats for the verified route
        try:
            length, elevation_gain, elevation_loss, grades = self._calculate_route_stats(verified_route)
            
            stats = {
                'length_km': length / 1000,
                'elevation_gain_m': elevation_gain,
                'elevation_loss_m': elevation_loss,
                'grades': grades,
                'quality_score': elevation_gain / (length/1000) if length > 0 else 0
            }
            
            self.route = verified_route
            self.route_stats = stats
            return verified_route, stats
        except Exception as e:
            st.error(f"Error calculating route statistics: {str(e)}")
            # Return minimal valid stats as fallback
            minimal_stats = {
                'length_km': total_distance / 1000,
                'elevation_gain_m': 0,
                'elevation_loss_m': 0,
                'grades': [],
                'quality_score': 0
            }
            self.route = verified_route
            self.route_stats = minimal_stats
            return verified_route, minimal_stats
    
    def _find_thorough_route(self) -> Tuple[List, Dict[str, float]]:
        """
        Find a route using the original thorough algorithm.
        Computationally expensive but may find higher quality routes.
        
        Returns:
            route: List of node IDs representing the optimal route
            stats: Dictionary containing route statistics
        """
        # Get node nearest to the input location as starting point
        if hasattr(self, 'location') and self.location:
            try:
                start_point = ox.geocode(self.location)
                start_node = ox.distance.nearest_nodes(
                    self.G, 
                    X=start_point[1],  # longitude
                    Y=start_point[0]   # latitude
                )
            except Exception as e:
                st.warning(f"Could not geocode starting location precisely: {str(e)}. Using graph center.")
                # Fallback: use center of the graph
                center_y = (max(node_data['y'] for _, node_data in self.G.nodes(data=True)) + 
                           min(node_data['y'] for _, node_data in self.G.nodes(data=True))) / 2
                center_x = (max(node_data['x'] for _, node_data in self.G.nodes(data=True)) + 
                           min(node_data['x'] for _, node_data in self.G.nodes(data=True))) / 2
                start_node = ox.distance.nearest_nodes(self.G, X=center_x, Y=center_y)
        else:
            # Fallback: use first node
            start_node = list(self.G.nodes)[0]
        
        # Find candidate paths that form loops
        st.info("Finding optimal elevation route (thorough search)...")
        
        # Strategy: Generate multiple routes and select the one with maximum elevation gain
        best_route = None
        best_gain = 0
        best_stats = {}
        
        # Calculate node elevations if not already present
        node_elevations = {}
        for node, data in self.G.nodes(data=True):
            node_elevations[node] = data.get('elevation', 0)
        
        # Enhanced topographic analysis: find areas with higher elevation variation
        elevation_variation = {}
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            if neighbors:
                neighbor_elevations = [node_elevations.get(neigh, 0) for neigh in neighbors]
                if neighbor_elevations:
                    elevation_variation[node] = max(neighbor_elevations) - min(neighbor_elevations)
                else:
                    elevation_variation[node] = 0
            else:
                elevation_variation[node] = 0
        
        # Sort nodes by elevation variation to prioritize high-variation areas
        target_nodes = sorted(
            [(node, variation) for node, variation in elevation_variation.items() if variation > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Limit to top nodes with highest variation plus some random nodes for diversity
        import random
        top_nodes = [node for node, _ in target_nodes[:100]]
        random_nodes = random.sample(list(self.G.nodes()), min(100, len(self.G.nodes())))
        candidate_targets = list(set(top_nodes + random_nodes))
        
        # Try different path lengths
        for length_factor in [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]:
            target_length = self.distance_km * 1000 * length_factor
            
            # Use custom weight that heavily favors uphill segments
            for edge in self.G.edges(data=True):
                # Default grade to 0 if not available
                grade = edge[2].get('grade', 0)
                
                # Enhanced weighting function that more aggressively prioritizes climbs
                if grade > 0:  # Uphill
                    # Exponential decrease in weight for uphill - steeper hills now much more preferable
                    weight = 1.0 / (1.0 + grade*3)  # Increased grade factor for more aggressive hill seeking
                elif grade < 0:  # Downhill
                    # Penalty for downhill proportional to steepness
                    weight = 2.0 - grade/5  # Adjusted to make downhills less preferable
                else:  # Flat
                    weight = 1.5  # Neutral weight for flat segments
                
                # Ensure weight is always positive and reasonable
                weight = max(0.05, min(10.0, weight))
                
                # Set custom weight - multiply by length to respect distance
                for edge_key in self.G[edge[0]][edge[1]]:
                    self.G[edge[0]][edge[1]][edge_key]['custom_weight'] = weight * edge[2]['length']
            
            # Try different candidates as endpoints
            for target_node in candidate_targets:
                if target_node == start_node:
                    continue
                    
                try:
                    # Find a route from start to target using elevation-optimized weights
                    route_out = nx.shortest_path(
                        self.G, 
                        source=start_node, 
                        target=target_node, 
                        weight='custom_weight'
                    )
                    
                    # If route is too short compared to target, skip
                    route_out_length = sum(self.G[u][v][0]['length'] for u, v in zip(route_out[:-1], route_out[1:]))
                    if route_out_length < target_length * 0.3:
                        continue
                    
                    # Create a modified graph to find diverse return path
                    temp_G = self.G.copy()
                    
                    # Heavily penalize reusing the same edges (avoid out-and-back routes)
                    for u, v in zip(route_out[:-1], route_out[1:]):
                        edge_data = temp_G.get_edge_data(u, v)
                        for edge_key in edge_data:
                            if 'custom_weight' in edge_data[edge_key]:
                                # 5x penalty for reusing outbound edges
                                temp_G[u][v][edge_key]['custom_weight'] *= 5.0
                    
                    # Find way back with modified weights - using A* for better performance
                    # We want the return path to maximize elevation too but with heavy penalty for reuse
                    try:
                        route_back = nx.astar_path(
                            temp_G, 
                            source=target_node, 
                            target=start_node,
                            heuristic=lambda u, v: nx.shortest_path_length(temp_G, u, v, weight='length'),
                            weight='custom_weight'
                        )
                    except nx.NetworkXNoPath:
                        # If A* fails, try regular Dijkstra
                        route_back = nx.shortest_path(
                            temp_G, 
                            source=target_node, 
                            target=start_node, 
                            weight='custom_weight'
                        )
                    
                    # Combine routes to form a loop
                    route = route_out + route_back[1:]
                    
                    # Calculate stats for this route
                    length, elevation_gain, elevation_loss, grades = self._calculate_route_stats(route)
                    
                    # Get elevation gain per km as a quality metric
                    gain_per_km = elevation_gain / (length/1000) if length > 0 else 0
                    
                    # Check if this route is within acceptable distance range and has better gain
                    distance_diff_pct = abs(length - self.distance_km * 1000) / (self.distance_km * 1000)
                    
                    # More precise distance matching with quality score that prioritizes elevation gain
                    quality_score = gain_per_km * (1 - min(0.5, distance_diff_pct))
                    
                    # Accept route if within 25% of target distance and better quality score
                    current_best_gain_per_km = best_gain / (best_stats.get('length_km', 1) * 1000) if best_stats.get('length_km', 0) > 0 else 0
                    current_best_quality = current_best_gain_per_km * (1 - min(0.5, abs(best_stats.get('length_km', 0) * 1000 - self.distance_km * 1000) / (self.distance_km * 1000)))
                    
                    if distance_diff_pct <= 0.25 and quality_score > current_best_quality:
                        best_route = route
                        best_gain = elevation_gain
                        best_stats = {
                            'length_km': length / 1000,
                            'elevation_gain_m': elevation_gain,
                            'elevation_loss_m': elevation_loss,
                            'grades': grades,
                            'quality_score': quality_score
                        }
                except (nx.NetworkXNoPath, KeyError) as e:
                    # Silently continue if no path exists or edge key is missing
                    continue
                except Exception as e:
                    # Log other exceptions but continue trying
                    import logging
                    logging.warning(f"Error finding path: {str(e)}")
                    continue
        
        if best_route is None:
            st.error("Could not find a suitable route. Try adjusting parameters or choosing a different location.")
            return None, {}
            
        self.route = best_route
        self.route_stats = best_stats
        return best_route, best_stats

    def _calculate_route_stats(self, route: List) -> Tuple[float, float, float, List[float]]:
        """
        Calculate statistics for a given route using both edge grades and node elevations
        for improved accuracy.
        
        Args:
            route: List of node IDs representing the route
            
        Returns:
            length: Total length in meters
            elevation_gain: Total elevation gain in meters
            elevation_loss: Total elevation loss in meters
            grades: List of grade values for each segment
        """
        length = 0
        elevation_gain = 0
        elevation_loss = 0
        grades = []

        # Validate route integrity
        valid_segments = []
        for i in range(len(route)-1):
            u, v = route[i], route[i+1]
            if self.G.has_edge(u, v):
                valid_segments.append((u, v))
            else:
                # Skip invalid segment but log warning
                st.warning(f"Missing edge between nodes {u} and {v} - skipping segment")
                # If needed, connect with shortest path
                try:
                    connecting_path = nx.shortest_path(self.G, u, v, weight='length')
                    if len(connecting_path) > 2:  # More than just u,v
                        # Add intermediate nodes
                        for j in range(1, len(connecting_path)-1):
                            route.insert(i+j, connecting_path[j])
                except nx.NetworkXNoPath:
                    pass  # No path available, continue with warning
        
        # Check if nodes have elevation data
        has_node_elevation = all('elevation' in self.G.nodes[node] for node in route[:min(10, len(route))])
        
        # Improved calculation with two methods: edge grades and direct node elevations
        if has_node_elevation:
            # Use actual node elevations for more precise calculations
            prev_elevation = self.G.nodes[route[0]].get('elevation', 0)
            
            for i, (u, v) in enumerate(zip(route[:-1], route[1:])):
                # Get edge data (there might be multiple edges between u and v)
                edge_data = min(self.G[u][v].values(), key=lambda x: x.get('length', float('inf')))
                
                # Accumulate length
                segment_length = edge_data.get('length', 0)
                length += segment_length
                
                # Get direct node elevations
                current_elevation = self.G.nodes[v].get('elevation', 0)
                elevation_change = current_elevation - prev_elevation
                
                # Get the reported grade and validate against calculated grade
                reported_grade = edge_data.get('grade', 0)
                calculated_grade = (elevation_change / segment_length * 100) if segment_length > 0 else 0
                
                # Use the most reasonable grade (handle inconsistencies in data)
                if abs(reported_grade - calculated_grade) > 15 and abs(calculated_grade) < 40:
                    # If reported grade seems unreasonable, use calculated
                    effective_grade = calculated_grade
                else:
                    # Otherwise trust the reported grade
                    effective_grade = reported_grade
                
                grades.append(effective_grade)
                
                # Accumulate gain or loss based on actual elevation change
                if elevation_change > 0:
                    elevation_gain += elevation_change
                else:
                    elevation_loss += abs(elevation_change)
                
                prev_elevation = current_elevation
        else:
            # Fallback to using grades if elevations not directly available
            for u, v in zip(route[:-1], route[1:]):
                # Get edge data (there might be multiple edges between u and v)
                edge_data = min(self.G[u][v].values(), key=lambda x: x.get('length', float('inf')))
                
                # Accumulate length
                segment_length = edge_data.get('length', 0)
                length += segment_length
                
                # Get grade and calculate elevation change
                grade = edge_data.get('grade', 0)
                grades.append(grade)
                
                # Calculate elevation change from grade
                elevation_change = grade * segment_length / 100
                
                # Accumulate gain or loss
                if elevation_change > 0:
                    elevation_gain += elevation_change
                else:
                    elevation_loss += abs(elevation_change)
        
        # Apply reasonable constraints to catch data anomalies
        if elevation_gain > length * 0.5:  # More than 50% grade average is suspicious
            st.warning("Unusually high elevation gain detected. Data may be inaccurate.")
            elevation_gain = min(elevation_gain, length * 0.3)  # Cap at 30% average grade
        
        return length, elevation_gain, elevation_loss, grades

    def visualize_route(self):
        """Visualize route with segment-level coloring based on grade."""
        if self.route is None:
            st.error("No route to visualize. Please find a route first.")
            return None
            
        # Validate route has sufficient nodes
        if len(self.route) < 2:
            st.error("Route has insufficient nodes for visualization.")
            return None
        
        # Extract route coordinates directly from nodes
        route_coords = []
        route_grades = []
        
        # Process route segments using direct node coordinates
        for idx, (u, v) in enumerate(zip(self.route[:-1], self.route[1:])):
            # Safely extract node coordinates
            if 'y' in self.G.nodes[u] and 'x' in self.G.nodes[u] and 'y' in self.G.nodes[v] and 'x' in self.G.nodes[v]:
                u_coords = (self.G.nodes[u]['y'], self.G.nodes[u]['x'])
                v_coords = (self.G.nodes[v]['y'], self.G.nodes[v]['x'])
                
                # Get edge grade with defensive error handling
                try:
                    edge_data = min(self.G[u][v].values(), key=lambda x: x.get('length', float('inf')))
                    grade = edge_data.get('grade', 0)
                except Exception as e:
                    st.warning(f"Error retrieving grade for segment ({u},{v}): {str(e)}")
                    grade = 0
                    
                # Always add the first coordinate
                if idx == 0 or len(route_coords) == 0:
                    route_coords.append(u_coords)
                    
                # Always add the second coordinate
                route_coords.append(v_coords)
                
                # Add corresponding grade for this segment
                route_grades.append(grade)
        
        # Verify we have sufficient data
        if len(route_coords) < 2 or len(route_grades) < 1:
            st.error("Insufficient route data for visualization")
            return None
            
        # Ensure grades array matches coordinate segments
        # One more coord than grade (n+1 points for n segments)
        if len(route_coords) != len(route_grades) + 1:
            # Fix by replicating last grade if needed
            while len(route_coords) > len(route_grades) + 1:
                route_grades.append(route_grades[-1] if route_grades else 0)
            # Trim excess coordinates if needed
            while len(route_coords) < len(route_grades) + 1:
                route_grades.pop()
        
        # Smooth grades for visualization
        smoothed_grades = route_grades
        if len(route_grades) >= ROLLING_MEAN_WINDOW:
            try:
                smoothed_grades = uniform_filter1d(
                    route_grades, 
                    size=ROLLING_MEAN_WINDOW, 
                    mode='nearest'
                )
            except Exception as e:
                st.warning(f"Grade smoothing failed: {str(e)}")
        
        # Create map centered at the first point of the route
        map_obj = folium.Map(
            location=route_coords[0],
            zoom_start=13,
            tiles="cartodbpositron"
        )
        
        # Create colormap with bounds checking
        norm = colors.Normalize(vmin=-10, vmax=10)
        cmap = cm.RdYlGn
        
        # Add route segments with color based on grade
        for i in range(len(route_coords) - 1):
            segment_coords = [route_coords[i], route_coords[i+1]]
            grade_idx = min(i, len(smoothed_grades) - 1)  # Prevent index errors
            grade = smoothed_grades[grade_idx]
            
            # Get color with error handling
            try:
                color = colors.rgb2hex(cmap(norm(grade)))
            except Exception:
                color = "#808080"  # Fallback to gray
            
            # Add line segment
            folium.PolyLine(
                segment_coords,
                color=color,
                weight=5,
                opacity=0.8,
                tooltip=f"Grade: {grade:.1f}%"
            ).add_to(map_obj)
        
        # Add markers for start/end
        folium.Marker(
            route_coords[0],
            tooltip="Start/End",
            icon=folium.Icon(color="green", icon="flag"),
        ).add_to(map_obj)
        
        # Add legend (unchanged)
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                    padding: 10px; border: 2px solid grey; border-radius: 5px">
            <p><b>Grade</b></p>
            <div style="display: flex; align-items: center;">
                <div style="background: linear-gradient(to right, #d73027, #fc8d59, #fee08b, #d9ef8b, #91cf60, #1a9850); 
                           width: 200px; height: 20px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; width: 200px;">
                <span>-10%</span>
                <span>0%</span>
                <span>10%</span>
            </div>
        </div>
        """
        map_obj.get_root().html.add_child(folium.Element(legend_html))
        
        return map_obj


def main():
    st.set_page_config(
        page_title="Elevation Route Finder",
        page_icon="🚵‍♂️",
        layout="wide"
    )
    st.title("🚵‍♂️ D+ - Si toi aussi tu veux des mollets musclay 🦵") 
    st.markdown(
        """
        Trouve des itinéraires de vélo ou de course à pied qui maximisent le d+
        """
    )
    
    
    # Create sidebar for inputs
    st.sidebar.header("Route Parameters")
    
    
    # Location input
    location = st.sidebar.text_input(
        "Starting location:",
        value="Piedmont, California, USA",
        help="Enter an address, city, or place name"
    )
    
        # Activity type selection
    activity_type = st.sidebar.radio(
        "Select activity type:",
        options=["cycling", "running"],
        index=0,
        key="activity_type"
    )
    
    # Dynamic distance range based on activity type
    if activity_type == "running":
        min_distance = 3
        max_distance = 50
        default_distance = 10
    else:  # cycling
        min_distance = 5
        max_distance = 100
        default_distance = 25
    
    # Distance selection with dynamic range
    distance_km = st.sidebar.slider(
        "Route distance (km):",
        min_value=min_distance,
        max_value=max_distance,
        value=default_distance,
        step=DISTANCE_STEP,
        help="Select your desired route distance in kilometers",
        key="distance_km"
    )
    
    # Add algorithm strategy selection
    strategy = st.sidebar.selectbox(
        "Route finding strategy:",
        options=["greedy", "optimized", "thorough"],
        index=1,  # Default to optimized
        help="Select algorithm strategy: greedy (fastest), optimized (balanced), or thorough (best quality but slowest)"
    )
    
    # Add strategy descriptions
    strategy_info = {
        "greedy": "Fastest option (15-30 seconds). Produces good routes by incrementally building a path that prioritizes uphill segments.",
        "optimized": "Balanced option (1-3 minutes). Uses intelligent sampling to find high-quality routes while maintaining reasonable performance.",
        "thorough": "Highest quality option (5-15+ minutes). Extensively searches for the optimal route with maximum elevation gain."
    }
    
    st.sidebar.info(strategy_info[strategy])
    
    # Create columns for displaying info and map
    col1, col2 = st.columns([1, 3])
    
    # Button to find route
    if st.sidebar.button("Find Optimal Route", type="primary"):
        if not location:
            st.error("Please enter a valid location.")
            return
            
        try:
            # Initialize optimizer with selected strategy
            optimizer = ElevationRouteOptimizer(
                activity_type=activity_type,
                location=location,
                distance_km=distance_km,
                strategy=strategy
            )
            
            # Fetch network and add elevation data
            with st.spinner("Fetching network and elevation data..."):
                optimizer.fetch_network()
            
            # Find optimal route
            with st.spinner(f"Finding optimal route using {strategy} strategy..."):
                route, stats = optimizer.find_optimal_route()
            
            if route is None:
                st.error("Could not find a suitable route. Try adjusting parameters.")
                return
                
            # Display route statistics
            with col1:
                st.subheader("Route Statistics")
                st.metric("Distance", f"{stats['length_km']:.2f} km")
                st.metric("Elevation Gain", f"{stats['elevation_gain_m']:.0f} m")
                st.metric("Elevation Loss", f"{stats['elevation_loss_m']:.0f} m")
                
                # Calculate and display average grade
                if stats['length_km'] > 0:
                    avg_grade = stats['elevation_gain_m'] / (stats['length_km'] * 1000) * 100

                    st.metric("Average Grade", f"{avg_grade:.1f}%")
                
                # Display quality score
                quality_score = stats.get('quality_score', 0)

                # Scale the score to a range of 1-10 legs
                # Assuming maximum quality score might be around 50-60
                # Adjust the scaling factor based on your actual observed range
                max_quality_score = 60  # Adjust based on your data
                leg_count = max(1, min(5, int(quality_score / max_quality_score * 5) + 1))
                
                # Create a string of leg emojis
                legs_display = "🦵" * leg_count
                
                # Display the quality as legs with the actual score in parentheses
                st.markdown(f"**Route Quality:** {legs_display}")            
            # Visualize route
            with col2:
                st.subheader("Route Map")
                map_obj = optimizer.visualize_route()
                folium_static(map_obj, width=800)
                
                # Add explanatory text
                st.markdown("""
                    **Map Legend:**
                    - Green segments: Uphill (positive grade)
                    - Yellow segments: Flat (near zero grade)
                    - Red segments: Downhill (negative grade)
                    - The darker the color, the steeper the grade
                """)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    
    # Add additional information
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### About
        This application uses:
        - OpenStreetMap (OSM) for street networks
        - Open Topo Data for elevation data
        - NetworkX for route optimization
        - Streamlit and Folium for visualization
        
        The algorithm attempts to find routes with maximum elevation gain 
        while keeping close to your desired distance.
        """
    )

if __name__ == "__main__":
    main()