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
from geopy.distance import geodesic
import heapq
from dataclasses import dataclass

if 'selected_climb_index' not in st.session_state:
    st.session_state.selected_climb_index = 0

# Configuration constants
MIN_DISTANCE_KM = 5
MAX_DISTANCE_KM = 150
DISTANCE_STEP = 5
SEARCH_RADIUS_BUFFER = 1.3  # Buffer multiplier for search radius
OPENTOPODATA_URL = "https://api.opentopodata.org/v1/eudem25m?locations={locations}"
BATCH_SIZE = 400
API_PAUSE = 0.1  # Pause between API calls in seconds
ROLLING_MEAN_WINDOW = 5  # Window size for smoothing grades

@dataclass
class ClimbSection:
    """Class to represent a climb section with relevant metrics."""
    edges: List[Tuple]  # List of (u, v, k) edge identifiers
    nodes: List  # List of node IDs that make up the path
    length_m: float  # Total length in meters
    elevation_gain_m: float  # Total elevation gain in meters
    avg_grade: float  # Average grade as percentage
    max_grade: float  # Maximum grade as percentage
    start_coords: Tuple[float, float]  # (lat, lon) of start
    end_coords: Tuple[float, float]  # (lat, lon) of end
    geometry: List[Tuple[float, float]]  # List of (lat, lon) coordinates along the path
    
    def get_score(self) -> float:
        """Calculate a score for ranking climbs based on length, grade and elevation gain."""
        # Score climbs by prioritizing longer, steeper sections with significant gain
        return (self.elevation_gain_m * (1 + self.avg_grade / 10))


class SteepClimbFinder:
    """Class to find steep, meaningful climbs within a specified area."""
    
    def __init__(self, 
                 location: str, 
                 radius_km: float, 
                 min_climb_length_m: float = 200, 
                 min_elevation_gain_m: float = 30,
                 min_avg_grade: float = 5.0,
                 max_results: int = 10):
        """
        Initialize the steep climb finder.
        
        Args:
            location: The starting location as a string
            radius_km: Search radius in kilometers
            min_climb_length_m: Minimum length of a climb in meters
            min_elevation_gain_m: Minimum elevation gain threshold in meters
            min_avg_grade: Minimum average grade to consider (percent)
            max_results: Maximum number of results to return
        """
        self.location = location
        self.radius_km = radius_km
        self.min_climb_length_m = min_climb_length_m
        self.min_elevation_gain_m = min_elevation_gain_m
        self.min_avg_grade = min_avg_grade
        self.max_results = max_results
        
        # Initialize with None; will be populated in fetch_network
        self.G = None
        self.center_point = None
        self.climbs = []
    
    def fetch_network(self) -> None:
        """Fetch the street network and add elevation data."""
        try:
            st.info(f"Fetching network around {self.location} with {self.radius_km}km radius...")
            
            # Get center point coordinates
            self.center_point = ox.geocode(self.location)
            
            # Convert radius to meters and fetch network
            radius_m = self.radius_km * 1000
            
            # For cycling network (prioritizing roads over paths)
            self.G = ox.graph_from_point(
                self.center_point, 
                dist=radius_m, 
                network_type="drive",  # Use "drive" to focus on roads
                simplify=True
            )
            
            st.success(f"Network fetched with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
            
            # Add elevation data
            try:
                # Try local elevation service first if available
                local_elevation_url = "http://localhost:5001/v1/eudem25m?locations={locations}"
                
                st.info("Adding elevation data...")
                try:
                    original_elevation_url = ox.settings.elevation_url_template
                    ox.settings.elevation_url_template = local_elevation_url
                    
                    self.G = ox.elevation.add_node_elevations_google(
                        self.G, 
                        api_key=None,
                        batch_size=500,
                        pause=0.01
                    )
                    st.success("Added elevation data from local service")
                    local_success = True
                except Exception as e:
                    st.warning(f"Local elevation service failed: {str(e)}. Trying public API...")
                    local_success = False
                    
                # If local service failed, try the public API
                if not local_success:
                    public_api_url = "https://api.opentopodata.org/v1/eudem25m?locations={locations}"
                    ox.settings.elevation_url_template = public_api_url
                    
                    self.G = ox.elevation.add_node_elevations_google(
                        self.G, 
                        api_key=None,
                        batch_size=100, 
                        pause=1.0
                    )
                    st.success("Added elevation data from public API")
                
                # Calculate edge grades
                self.G = ox.elevation.add_edge_grades(self.G)
                
                # Verify elevation data quality
                elevations = [data.get('elevation', 0) for _, data in self.G.nodes(data=True)]
                if elevations:
                    min_elev = min(elevations)
                    max_elev = max(elevations)
                    elevation_range = max_elev - min_elev
                    
                    if elevation_range < 10.0:
                        st.warning(f"Very flat terrain detected (elevation range: {elevation_range:.1f}m). " 
                                  f"May be difficult to find steep climbs.")
                    else:
                        st.info(f"Terrain elevation range: {min_elev:.1f}m to {max_elev:.1f}m " 
                               f"(range: {elevation_range:.1f}m)")
                
                # Restore original settings if modified
                if 'original_elevation_url' in locals():
                    ox.settings.elevation_url_template = original_elevation_url
                    
            except Exception as e:
                st.error(f"Error adding elevation data: {str(e)}")
                raise
                
        except Exception as e:
            st.error(f"Error fetching network: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def find_climbs(self) -> List[ClimbSection]:
        """
        Find steep climbs that meet the criteria with improved overlap handling.
        
        Returns:
            List of ClimbSection objects representing the identified climbs
        """
        if self.G is None:
            st.error("Network not fetched. Please fetch network first.")
            return []
        
        st.info("Analyzing network for steep climbs...")
        
        # 1. Find uphill edges with sufficient grade
        steep_edges = []
        
        # First pass: identify candidate edges
        for u, v, k, data in self.G.edges(keys=True, data=True):
            # Skip edges without grade data
            if 'grade' not in data:
                continue
                
            grade = data['grade'] * 100 
            length = data.get('length', 0)
            
            # Collect uphill edges meeting minimum grade
            if grade >= self.min_avg_grade and length > 0:
                # Store as (u, v, k) with metadata
                steep_edges.append((u, v, k, {
                    'grade': grade,
                    'length': length,
                    'elevation_gain': length * grade / 100
                }))
        
        # If no steep edges found, return early
        if not steep_edges:
            st.warning(f"No edges with grades above {self.min_avg_grade}% found in this area.")
            return []
            
        # 2. Find connected components/sequences of steep edges
        # Sort edges by node ID to help identify connected segments
        steep_edges.sort()
        
        # Create a specialized graph of just the steep edges
        steep_graph = nx.DiGraph()
        
        # Add edges to the steep graph
        for u, v, k, data in steep_edges:
            steep_graph.add_edge(u, v, key=k, **data)
            
            # Add node attributes using data from original graph
            steep_graph.nodes[u].update({
                'x': self.G.nodes[u].get('x'),
                'y': self.G.nodes[u].get('y'),
                'elevation': self.G.nodes[u].get('elevation', 0)
            })
            steep_graph.nodes[v].update({
                'x': self.G.nodes[v].get('x'),
                'y': self.G.nodes[v].get('y'),
                'elevation': self.G.nodes[v].get('elevation', 0)
            })
        
        # 3. Identify climb paths using depth-first traversal
        climb_paths = []
        
        # Visit each node as a potential start of a climb
        for start_node in steep_graph.nodes():
            # Skip nodes with no outgoing edges
            if steep_graph.out_degree(start_node) == 0:
                continue
                
            # Perform depth-first traversal to find climb paths
            self._find_climb_paths_from_node(
                steep_graph, 
                start_node, 
                [], 
                0, 
                0, 
                0, 
                set(),
                climb_paths
            )
        
        # 4. Score, filter, and deduplicate paths
        scored_paths = []
        
        # First filter based on minimum requirements and calculate scores
        for path, data in climb_paths:
            # Skip if path doesn't meet minimum requirements
            if (data['length'] < self.min_climb_length_m or 
                data['elevation_gain'] < self.min_elevation_gain_m or
                data['avg_grade'] < self.min_avg_grade):
                continue
                
            # Calculate quality score for this climb
            # Prioritize longer climbs with higher grades and elevation gain
            score = data['elevation_gain'] * (1 + data['avg_grade'] / 10) * (data['length'] / 1000)
            
            # Store for later processing
            scored_paths.append((path, data, score))
        
        # No valid paths after filtering
        if not scored_paths:
            st.warning(f"No climbs meeting the minimum criteria found. Try adjusting parameters.")
            return []
            
        # Sort by score (descending)
        scored_paths.sort(key=lambda x: x[2], reverse=True)
        
        # 5. Process paths to avoid excessive overlap
        climbs = []
        processed_segments = set()  # Track already included segments
        
        for path, data, score in scored_paths:
            # Create a set of edge tuples representing this path
            path_segments = set()
            for i in range(len(path) - 1):
                # Create a segment identifier (u, v)
                path_segments.add((path[i], path[i+1]))
            
            # Calculate overlap with already processed segments
            if processed_segments:
                # Number of overlapping segments
                overlap_count = len(path_segments.intersection(processed_segments))
                # Percentage of this path that overlaps with existing paths
                overlap_percentage = overlap_count / len(path_segments)
                
                # Skip if overlap is too high (adjustable threshold)
                if overlap_percentage > 0.4:  # 40% overlap threshold
                    continue
            
            # Path passed overlap check, extract geometry and prepare climb object
            # Get coordinates for visualization
            geometry = []
            try:
                # Try to get detailed geometry if available
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge_data = steep_graph[u][v]
                    
                    if 'geometry' in edge_data:
                        # LineString from OSMnx has (lon, lat) ordering
                        geom = edge_data['geometry']
                        # Convert to (lat, lon) for Folium
                        coords = [(lat, lon) for lon, lat in geom.coords]
                        geometry.extend(coords)
                    else:
                        # Fallback to node coordinates
                        geometry.append((steep_graph.nodes[u]['y'], steep_graph.nodes[u]['x']))
                        geometry.append((steep_graph.nodes[v]['y'], steep_graph.nodes[v]['x']))
            except Exception:
                # Fallback: use node coordinates directly
                geometry = [(steep_graph.nodes[n]['y'], steep_graph.nodes[n]['x']) for n in path]
            
            # Remove duplicate consecutive points
            clean_geometry = []
            for point in geometry:
                if not clean_geometry or point != clean_geometry[-1]:
                    clean_geometry.append(point)
            
            # Create start and end coordinates
            if clean_geometry:
                start_coords = clean_geometry[0]
                end_coords = clean_geometry[-1]
            else:
                # Fallback
                start_coords = (steep_graph.nodes[path[0]]['y'], steep_graph.nodes[path[0]]['x'])
                end_coords = (steep_graph.nodes[path[-1]]['y'], steep_graph.nodes[path[-1]]['x'])
            
            # Create edge identification tuples
            edges = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Get the key (k) for this edge
                for k in steep_graph[u][v]:
                    edges.append((u, v, k))
                    break
            
            # Create ClimbSection object
            climb = ClimbSection(
                edges=edges,
                nodes=path,
                length_m=data['length'],
                elevation_gain_m=data['elevation_gain'],
                avg_grade=data['avg_grade'],
                max_grade=data['max_grade'],
                start_coords=start_coords,
                end_coords=end_coords,
                geometry=clean_geometry
            )
            
            # Add to results and update processed segments
            climbs.append(climb)
            processed_segments.update(path_segments)
            
            # Limit to max_results
            if len(climbs) >= self.max_results:
                break
        
        # Final sort of selected climbs by score
        climbs.sort(key=lambda x: x.get_score(), reverse=True)
        
        st.success(f"Found {len(climbs)} significant climbs with minimal overlap")
        self.climbs = climbs
        return climbs
    
    def _find_climb_paths_from_node(self, 
                                    graph: nx.DiGraph, 
                                    current: Any, 
                                    path: List, 
                                    length: float, 
                                    gain: float, 
                                    max_grade: float,
                                    visited: set,
                                    results: List) -> None:
        """
        Recursively find climb paths from a starting node using depth-first traversal.
        
        Args:
            graph: NetworkX graph of steep edges
            current: Current node
            path: Current path (list of nodes)
            length: Accumulated path length
            gain: Accumulated elevation gain
            max_grade: Maximum grade encountered
            visited: Set of visited nodes (to prevent cycles)
            results: List to collect results
        """
        # Add current node to path
        path = path + [current]
        visited = visited.copy()
        visited.add(current)
        
        # If path has more than one node, it's a potential climb
        if len(path) > 1:
            # Calculate average grade for the current path
            avg_grade = (gain / length * 100) if length > 0 else 0
            
            # Record as a potential climb if it meets minimum requirements
            if (length >= self.min_climb_length_m and 
                gain >= self.min_elevation_gain_m and 
                avg_grade >= self.min_avg_grade):
                
                # Store as (path, metadata)
                results.append((path.copy(), {
                    'length': length,
                    'elevation_gain': gain,
                    'avg_grade': avg_grade,
                    'max_grade': max_grade
                }))
        
        # Get neighbor nodes (outgoing edges)
        for neighbor in graph.successors(current):
            # Skip if already visited (avoid cycles)
            if neighbor in visited:
                continue
                
            # Get edge data
            edge_data = graph[current][neighbor]
            
            # OSMnx might have multiple edges between same nodes, get first one
            if isinstance(edge_data, dict) and not isinstance(edge_data, nx.classes.coreviews.AtlasView):
                # Single edge
                edge = edge_data
            else:
                # Multiple edges, get the one with highest grade
                edge = max(edge_data.values(), key=lambda x: x.get('grade', 0))
            
            # Get edge attributes
            edge_length = edge.get('length', 0)
            edge_grade = edge.get('grade', 0)
            edge_gain = edge_length * edge_grade / 100
            
            # Update max grade
            new_max_grade = max(max_grade, edge_grade)
            
            # Only continue if this edge is uphill
            if edge_grade >= self.min_avg_grade:
                # Recursively explore from neighbor
                self._find_climb_paths_from_node(
                    graph,
                    neighbor,
                    path,
                    length + edge_length,
                    gain + edge_gain,
                    new_max_grade,
                    visited,
                    results
                )
    
    def visualize_climbs(self) -> folium.Map:
        """
        Create a visualization of the identified climbs that follows actual road geometry.
        
        Returns:
            folium.Map object with visualized climbs
        """
        if not self.climbs:
            st.warning("No climbs to visualize. Run find_climbs() first.")
            return None
        
        # Find center point from original center or first climb
        if self.center_point:
            map_center = (self.center_point[0], self.center_point[1])
        elif self.climbs:
            map_center = self.climbs[0].start_coords
        else:
            # Fallback to a default center from network
            nodes_df = ox.graph_to_gdfs(self.G, edges=False)
            map_center = (nodes_df['y'].mean(), nodes_df['x'].mean())
        
        # Create map
        m = folium.Map(
            location=map_center,
            zoom_start=14,
            tiles="cartodbpositron"
        )
        
        # Create colormap for climbs
        cmap = cm.get_cmap('plasma', len(self.climbs))
        
        # Add circle marker for center point
        folium.CircleMarker(
            location=map_center,
            radius=10,
            color='blue',
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Center: {self.location}"
        ).add_to(m)
        
        # Add circles to show search radius
        folium.Circle(
            location=map_center,
            radius=self.radius_km * 1000,  # Convert to meters
            color='blue',
            weight=2,
            fill=False,
            opacity=0.5,
            tooltip=f"Search radius: {self.radius_km}km"
        ).add_to(m)
        
        # Add each climb with a different color
        for i, climb in enumerate(self.climbs):
            # Generate color for this climb
            color = colors.rgb2hex(cmap(i / len(self.climbs)))
            
            # Collect detailed road geometry for this climb
            detailed_geometry = []
            
            for j in range(len(climb.nodes) - 1):
                u, v = climb.nodes[j], climb.nodes[j+1]
                
                # Get the edge data between these nodes
                if self.G.has_edge(u, v):
                    # There might be multiple edges between u and v, get all of them
                    edge_data = self.G[u][v]
                    
                    # If there are multiple edges, find the one with the right key
                    # or just take the first one if we can't find a perfect match
                    edge_geometry = None
                    
                    # Try to find an edge with geometry data
                    for k, data in edge_data.items():
                        if 'geometry' in data:
                            # Found edge with geometry
                            geom = data['geometry']
                            # Convert to (lat, lon) for Folium (OSMnx stores as lon, lat)
                            coords = [(lat, lon) for lon, lat in geom.coords]
                            edge_geometry = coords
                            break
                    
                    if edge_geometry:
                        # Add the detailed geometry
                        detailed_geometry.extend(edge_geometry)
                    else:
                        # Fallback to node coordinates if no geometry found
                        detailed_geometry.append((self.G.nodes[u]['y'], self.G.nodes[u]['x']))
                        detailed_geometry.append((self.G.nodes[v]['y'], self.G.nodes[v]['x']))
                else:
                    # Edge not found, use node coordinates
                    detailed_geometry.append((self.G.nodes[u]['y'], self.G.nodes[u]['x']))
                    detailed_geometry.append((self.G.nodes[v]['y'], self.G.nodes[v]['x']))
            
            # Use the detailed geometry if available, otherwise fall back to the original
            geometry_to_use = detailed_geometry if detailed_geometry else climb.geometry
            
            # Remove duplicate consecutive points
            clean_geometry = []
            for point in geometry_to_use:
                if not clean_geometry or point != clean_geometry[-1]:
                    clean_geometry.append(point)
            
            # Add polyline for climb
            folium.PolyLine(
                clean_geometry,
                color=color,
                weight=5,
                opacity=0.8,
                tooltip=f"Climb {i+1}: {climb.length_m:.0f}m, {climb.avg_grade:.1f}%, {climb.elevation_gain_m:.0f}m gain"
            ).add_to(m)
            
            # Add markers for start and end
            folium.Marker(
                climb.start_coords,
                icon=folium.Icon(icon='play', color='green'),
                tooltip=f"Start of climb {i+1}"
            ).add_to(m)
            
            folium.Marker(
                climb.end_coords,
                icon=folium.Icon(icon='flag', color='red'),
                tooltip=f"End of climb {i+1}"
            ).add_to(m)
            
            # Add a label with climb number
            folium.map.Marker(
                climb.start_coords,
                icon=DivIcon(
                    icon_size=(20, 20),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size: 12pt; color: white; background-color: {color}; border-radius: 50%; width: 25px; height: 25px; text-align: center; line-height: 25px;">{i+1}</div>'
                )
            ).add_to(m)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                    padding: 10px; border: 2px solid grey; border-radius: 5px">
            <p><b>Climbs by Rank</b></p>
        """
        
        # Add entry for each climb
        for i, climb in enumerate(self.climbs):
            color = colors.rgb2hex(cmap(i / len(self.climbs)))
            legend_html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <span style="background-color: {color}; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span>
                <span>#{i+1}: {climb.length_m:.0f}m @ {climb.avg_grade:.1f}% ({climb.elevation_gain_m:.0f}m gain)</span>
            </div>
            """
            
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m

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
                buffer_factor = 1.4  # Larger buffer for short routes
            elif self.distance_km < 20:
                buffer_factor = 1.2
            elif self.distance_km < 50:
                buffer_factor = 1
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
        """Visualize route with segment-level coloring based on grade, following actual road geometry."""
        if self.route is None:
            st.error("No route to visualize. Please find a route first.")
            return None
            
        # Validate route has sufficient nodes
        if len(self.route) < 2:
            st.error("Route has insufficient nodes for visualization.")
            return None
        
        # Extract route coordinates and grades
        route_segments = []  # Will store tuples of (coordinates, grade)
        
        # Process route segments to extract detailed geometry
        for idx, (u, v) in enumerate(zip(self.route[:-1], self.route[1:])):
            # Skip if nodes don't exist in graph
            if not self.G.has_edge(u, v):
                continue
                
            # Get all edges between these nodes (could be multiple)
            edge_data_options = self.G[u][v]
            
            # Default to direct coordinates if we can't find geometry
            u_coords = (self.G.nodes[u]['y'], self.G.nodes[u]['x'])
            v_coords = (self.G.nodes[v]['y'], self.G.nodes[v]['x'])
            
            # Find the edge with best data (prioritize ones with geometry)
            best_edge_data = None
            best_key = None
            
            for k, data in edge_data_options.items():
                if best_edge_data is None or 'geometry' in data:
                    best_edge_data = data
                    best_key = k
            
            # Extract grade for this segment
            grade = best_edge_data.get('grade', 0)
            
            # Get detailed geometry if available
            segment_coords = []
            if 'geometry' in best_edge_data:
                # LineString geometry from OSMnx has (lon, lat) ordering
                geom = best_edge_data['geometry']
                # Convert to (lat, lon) for Folium
                segment_coords = [(lat, lon) for lon, lat in geom.coords]
            else:
                # Fallback to simple straight line
                segment_coords = [u_coords, v_coords]
            
            # Store the segment with its grade
            route_segments.append((segment_coords, grade))
        
        # Verify we have sufficient data
        if not route_segments:
            st.error("No valid segments found for visualization")
            return None
        
        # Extract starting point for map centering
        first_segment = route_segments[0]
        first_point = first_segment[0][0] if first_segment[0] else (
            self.G.nodes[self.route[0]]['y'], 
            self.G.nodes[self.route[0]]['x']
        )
        
        # Create map centered at the first point
        map_obj = folium.Map(
            location=first_point,
            zoom_start=13,
            tiles="cartodbpositron"
        )
        
        # Create colormap for grade visualization
        norm = colors.Normalize(vmin=-10, vmax=10)
        cmap = cm.RdYlGn
        
        # Add segments with color based on grade
        for segment_coords, grade in route_segments:
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
            first_point,
            tooltip="Start/End",
            icon=folium.Icon(color="green", icon="flag"),
        ).add_to(map_obj)
        
        # Add legend
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


# Helper functions for the steep climb finder functionality
def get_mime_type(export_format):
    """Return the appropriate MIME type for download button."""
    mime_types = {
        "GPX": "application/gpx+xml",
        "CSV": "text/csv",
        "GeoJSON": "application/geo+json"
    }
    return mime_types.get(export_format, "text/plain")

def prepare_export_data(climbs, format_type):
    """Prepare climb data for export in various formats."""
    if format_type == "GPX":
        # Create GPX file with each climb as a track
        gpx_data = '<?xml version="1.0" encoding="UTF-8"?>\n'
        gpx_data += '<gpx version="1.1" creator="D+ Steep Climb Finder" xmlns="http://www.topografix.com/GPX/1/1">\n'
        
        for i, climb in enumerate(climbs):
            gpx_data += f'  <trk>\n    <name>Climb {i+1}: {climb.length_m:.0f}m @ {climb.avg_grade:.1f}%</name>\n    <trkseg>\n'
            
            # Add trackpoints
            for lat, lon in climb.geometry:
                gpx_data += f'      <trkpt lat="{lat}" lon="{lon}"></trkpt>\n'
                
            gpx_data += '    </trkseg>\n  </trk>\n'
            
        gpx_data += '</gpx>'
        return gpx_data
        
    elif format_type == "CSV":
        # Create CSV with climb details
        csv_data = "Climb,Length (m),Elevation Gain (m),Average Grade (%),Maximum Grade (%),Start Latitude,Start Longitude,End Latitude,End Longitude\n"
        
        for i, climb in enumerate(climbs):
            csv_data += f"{i+1},{climb.length_m:.1f},{climb.elevation_gain_m:.1f},{climb.avg_grade:.2f},{climb.max_grade:.2f},{climb.start_coords[0]},{climb.start_coords[1]},{climb.end_coords[0]},{climb.end_coords[1]}\n"
            
        return csv_data
        
    elif format_type == "GeoJSON":
        # Create GeoJSON with each climb as a feature
        geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }
        
        for i, climb in enumerate(climbs):
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lat, lon in climb.geometry]  # GeoJSON uses [lon, lat] order
                },
                "properties": {
                    "id": i + 1,
                    "name": f"Climb {i+1}",
                    "length_m": round(climb.length_m, 1),
                    "elevation_gain_m": round(climb.elevation_gain_m, 1),
                    "avg_grade": round(climb.avg_grade, 2),
                    "max_grade": round(climb.max_grade, 2)
                }
            }
            geojson_data["features"].append(feature)
            
        import json
        return json.dumps(geojson_data, indent=2)
    
    return ""

def generate_elevation_profile(climb, graph):
    """Generate an elevation profile chart for a climb."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    
    # Extract node elevations along the path
    elevations = []
    distances = []
    cumulative_distance = 0
    
    # Get elevation and distance data
    for i in range(len(climb.nodes) - 1):
        u, v = climb.nodes[i], climb.nodes[i + 1]
        
        # Get node elevations
        u_elev = graph.nodes[u].get('elevation', 0)
        v_elev = graph.nodes[v].get('elevation', 0)
        
        # Get edge length - find the right edge if multiple exist
        edge_data = min(graph[u][v].values(), key=lambda x: x.get('length', float('inf')))
        edge_length = edge_data.get('length', 0)
        
        # Add start point if this is the first segment
        if i == 0:
            elevations.append(u_elev)
            distances.append(cumulative_distance)
            
        # Add end point
        elevations.append(v_elev)
        cumulative_distance += edge_length
        distances.append(cumulative_distance)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot elevation profile
    ax.plot(distances, elevations, 'b-', linewidth=2.5)
    ax.fill_between(distances, elevations, min(elevations), alpha=0.3, color='skyblue')
    
    # Add labels and grid
    ax.set_xlabel('Distance (meters)')
    ax.set_ylabel('Elevation (meters)')
    ax.set_title(f'Elevation Profile: {climb.length_m:.0f}m @ {climb.avg_grade:.1f}%, {climb.elevation_gain_m:.0f}m gain')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add key stats as text annotation
    stats_text = (
        f"Length: {climb.length_m:.0f}m\n"
        f"Elevation Gain: {climb.elevation_gain_m:.0f}m\n"
        f"Average Grade: {climb.avg_grade:.1f}%\n"
        f"Maximum Grade: {climb.max_grade:.1f}%"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set y-axis limits with some padding
    min_elev = min(elevations)
    max_elev = max(elevations)
    y_range = max_elev - min_elev
    ax.set_ylim(min_elev - 0.1 * y_range, max_elev + 0.1 * y_range)
    
    # Add horizontal distance markers every 100m
    marker_interval = 100  # meters
    for d in range(0, int(max(distances)), marker_interval):
        if d <= max(distances):
            ax.axvline(x=d, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    return fig

def main():
    # Tab-specific session state initialization
    if 'climb_results' not in st.session_state:
        st.session_state.climb_results = []

    st.set_page_config(
        page_title="Elevation Route Finder",
        page_icon="🚵‍♂️",
        layout="wide"
    )
    st.title("🚵‍♂️ D+ - Si toi aussi tu veux des mollets musclay 🦵") 
    
    # Create tabs for different app functions
    tab1, tab2 = st.tabs(["Route Finder", "Steep Climb Finder"])
    
    with tab1:
        # Route Finder tab-specific session state
        if 'route_results' not in st.session_state:
            st.session_state.route_results = None
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
            value="Quai Gustave Ador 23, 1207 Genève",
            help="Enter an address, city, or place name",
            key="route_location"
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
        if st.sidebar.button("Find Optimal Route", type="primary", key="find_route_button"):
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
                    leg_count = max(1, min(5, int(quality_score / 60 * 5) + 1))
                    legs_display = "🦵" * leg_count
                    st.markdown(f"**Route Quality:** {legs_display}")            
                
                # Visualize route
                with col2:
                    st.subheader("Route Map")
                    map_obj = optimizer.visualize_route()
                    folium_static(map_obj, width=800, height=500)
                    
                    # Add option to download map as HTML
                    if st.button("Export Map as HTML"):
                        html_data = map_obj._repr_html_()
                        st.download_button(
                            "Download HTML Map",
                            html_data,
                            file_name=f"route_map_{location.replace(' ', '_')}.html",
                            mime="text/html"
                        )
                    
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
    
    # Steep Climb Finder Tab
    with tab2:
        st.markdown(
            """
            ## Steep Climb Finder
            Discover significant uphill segments within a specified radius. Perfect for finding the steepest, 
            most challenging climbs near you for training.
            """
        )
        
        # Create sidebar for steep climb inputs
        st.sidebar.header("Climb Finder Parameters")
        
        # Location input
        climb_location = st.sidebar.text_input(
            "Starting location:",
            value="Quai Gustave Ador 23, 1207 Genève",
            help="Enter an address, city, or place name",
            key="climb_location"
        )
        
        # Search radius
        radius_km = st.sidebar.slider(
            "Search radius (km):",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Radius around the starting point to search for climbs",
            key="radius_km"
        )
        
        # Advanced parameters
        with st.sidebar.expander("Advanced Parameters"):
            min_climb_length_m = st.slider(
                "Minimum climb length (m):",
                min_value=100,
                max_value=1000,
                value=200,
                step=50,
                help="Minimum length of a climb section to be considered"
            )
            
            min_elevation_gain_m = st.slider(
                "Minimum elevation gain (m):",
                min_value=20,
                max_value=100,
                value=30,
                step=5,
                help="Minimum elevation gain for a climb to be considered significant"
            )
            
            min_avg_grade = st.slider(
                "Minimum average grade (%):",
                min_value=3.0,
                max_value=15.0,
                value=5.0,
                step=0.5,
                help="Minimum average grade for a climb to be considered"
            )
            
            max_results = st.slider(
                "Maximum results:",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                help="Maximum number of climbs to return"
            )
        
        # Create columns for climb info and map
        climb_col1, climb_col2 = st.columns([1, 3])
        
        # Button to find climbs
        if st.sidebar.button("Find Steep Climbs", type="primary", key="find_climbs_button"):
            if not climb_location:
                st.error("Please enter a valid location.")
                return
                
            try:
                # Initialize steep climb finder
                climb_finder = SteepClimbFinder(
                    location=climb_location,
                    radius_km=radius_km,
                    min_climb_length_m=min_climb_length_m,
                    min_elevation_gain_m=min_elevation_gain_m,
                    min_avg_grade=min_avg_grade,
                    max_results=max_results
                )
                
                # Fetch network and add elevation data
                with st.spinner("Fetching network and elevation data..."):
                    climb_finder.fetch_network()
                
                # Find steep climbs
                with st.spinner("Analyzing network for steep climbs..."):
                    climbs = climb_finder.find_climbs()
                
                if not climbs:
                    st.warning("No significant climbs found that match your criteria. Try adjusting parameters or choosing a different location.")
                    return

                with climb_col2:
                    st.subheader("Climb Map")
                    
                    # Generate the map visualization
                    climb_map = climb_finder.visualize_climbs()
                    
                    # If a climb is selected, center the map on it
                    if hasattr(st.session_state, 'selected_climb_index') and st.session_state.selected_climb_index < len(climbs):
                        selected_climb = climbs[st.session_state.selected_climb_index]
                        # Find midpoint to center on
                        mid_idx = len(selected_climb.geometry) // 2
                        if mid_idx < len(selected_climb.geometry):
                            # Update map center to selected climb
                            climb_map.location = selected_climb.geometry[mid_idx]
                            # You might need to adjust zoom level too
                            climb_map.zoom_start = 15
                            
                    # Show the map
                    folium_static(climb_map, width=800, height=600)
                    
                # Create a variable to track selected climb for map centering
                # if 'selected_climb_index' not in st.session_state:
                #     st.session_state.selected_climb_index = 0
                
                # Display visualization controls above the columns
                climb_sort = st.radio(
                    "Sort climbs by:",
                    options=["Score (default)", "Length", "Grade", "Elevation Gain"],
                    horizontal=True,
                    key="climb_sort"
                )
                
                # Sort climbs based on selection
                if climb_sort == "Length":
                    climbs.sort(key=lambda x: x.length_m, reverse=True)
                elif climb_sort == "Grade":
                    climbs.sort(key=lambda x: x.avg_grade, reverse=True)
                elif climb_sort == "Elevation Gain":
                    climbs.sort(key=lambda x: x.elevation_gain_m, reverse=True)

                # "Score" is default, already sorted
                
                # Export options
                export_format = st.selectbox(
                    "Export climbs as:",
                    options=["None", "GPX", "CSV", "GeoJSON"],
                    index=0,
                    key="export_format"
                )
                
                if export_format != "None":
                    export_data = prepare_export_data(climbs, export_format)
                    st.download_button(
                        f"Download {export_format} file",
                        export_data,
                        file_name=f"steep_climbs_{climb_location.replace(' ', '_')}.{export_format.lower()}",
                        mime=get_mime_type(export_format)
                    )
                
                # Display climb statistics
                with climb_col1:
                    st.subheader("Climb Statistics")
                    
                    # Summary table of all climbs
                    climb_data = []
                    for i, climb in enumerate(climbs):
                        climb_data.append({
                            "Rank": i+1,
                            "Length (m)": f"{climb.length_m:.0f}",
                            "Gain (m)": f"{climb.elevation_gain_m:.0f}",
                            "Avg Grade (%)": f"{climb.avg_grade:.1f}",
                            "Max Grade (%)": f"{climb.max_grade:.1f}"
                        })
                    
                    st.dataframe(
                        climb_data, 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("---")
                    st.markdown("### Detailed Climb Information")
                    
                    # Create an expandable section for each climb
                    for i, climb in enumerate(climbs):
                        with st.expander(f"Climb #{i+1}: {climb.length_m:.0f}m @ {climb.avg_grade:.1f}%"):
                            st.metric("Distance", f"{climb.length_m:.0f} m")
                            st.metric("Elevation Gain", f"{climb.elevation_gain_m:.0f} m")
                            st.metric("Average Grade", f"{climb.avg_grade:.1f}%")
                            st.metric("Maximum Grade", f"{climb.max_grade:.1f}%")
                            
                            # Calculate difficulty score based on length, grade, and elevation gain
                            difficulty_score = climb.get_score()
                            scaled_score = min(5, max(1, int(difficulty_score / 200)))
                            fire_emoji = "🔥" * scaled_score
                            st.markdown(f"**Difficulty:** {fire_emoji}")
                            
                            # Add button to center map on this climb
                            if st.button(f"Center map on this climb", key=f"center_climb_{i}"):
                                st.session_state.selected_climb_index = i
                                # You may need to set a flag to trigger map redraw
                                st.session_state.redraw_map = True
            except Exception as e:
                # Handle the exception
                st.error(f"An error occurred: {str(e)}")
if __name__ == "__main__":
    main()