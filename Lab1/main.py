import sys
from queue import PriorityQueue

from maze import Maze, path_from

def L1 (a,b):
  return abs(a.x-b.x)+abs(a.y-b.y)

def find_minA(nodes):
  node=nodes[0]
  for n in nodes:
    if n.priority<node.priority:
      node=n
  return node

def find_min(nodes):
  node = nodes[0]
  for n in nodes:
    if n.cost<node.cost:
      node=n
  return node

def bfs(maze):
    start_node = maze.find_node('S')
    q = [start_node]
    while len(q) > 0:
        node = q.pop(0)  # FIFO
        node.visited = True
        if node.type == 'E':
            return path_from(node)

        children = maze.get_possible_movements(node)
        for child in children:
            if not child.visited:
                child.parent = node
                q.append(child)

    return None

  
def best_bfs(maze):
  start_node = maze.find_node('S')
  end_node = maze.find_node('E')
  start_node.cost=L1(start_node,end_node)
  queue = [start_node]
  while len(queue) > 0:
      #node = queue.pop(0)
      node = find_minA(queue)
      queue.remove(node)
      node.visited = True
      if node.type == 'E':
          return path_from(node)

      children = maze.get_possible_movements(node)
    
      for child in children:
          if not child.visited:
            child.parent = node
            child.priority=L1(node,child)  
            queue.append(child)

  return None


def Dijkstra(maze):
  start_node = maze.find_node('S')
  start_node.cost=0
  queue = [start_node]
  while len(queue) > 0:
    node = find_min(queue)
    queue.remove(node)
    node.visited=True

    if node.type == 'E':
            return path_from(node)
      
    children = maze.get_possible_movements(node) 
    
    for child in children:
      new_cost = node.cost + maze.move_cost(node,child)
      
      if new_cost< child.cost:
        child.cost=new_cost
        child.parent=node
        queue.append(child)
  
  return None

def Astar(maze):
    start_node = maze.find_node('S')
    end_node = maze.find_node('E')
    start_node.cost=L1(start_node,end_node)
    start_node.priority=0
    queue = [start_node]
    while len(queue) > 0:
      #node = queue.pop(0)

      node = find_minA(queue)
      queue.remove(node)

      node.visited=True
  
      if node.type == 'E':
              return path_from(node)
        
      children = maze.get_possible_movements(node) 
      
      for child in children:
        
        new_cost = node.cost + maze.move_cost(node,child)
        new_priority= new_cost+L1(child,node)
        
        if new_cost< child.cost:
          child.cost=new_cost
          child.parent=node
          child.priority=new_priority
          queue.append(child)
    
    return None


  

maze = Maze.from_file("maze3.txt")
maze.draw()

#maze.path = Dijkstra(maze)
#maze.path = Astar(maze)
#maze.path = bfs(maze)
# maze.path = best_bfs(maze)

# print()
# maze.draw()
# print('path length: ', len(maze.path))
# for node in maze.path:
#     print(f'({node.x}, {node.y})', end=' ')
# print()

maze.path = Astar(maze)

print()
maze.draw()
print('path length: ', len(maze.path))
for node in maze.path:
    print(f'({node.x}, {node.y})', end=' ')
print()

