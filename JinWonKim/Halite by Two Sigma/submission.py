
from kaggle_environments.envs.halite.helpers import *
from random import choice, randint

shipeye = 1
shipyardeye = 1


def getDirTo(fromPos, toPos, size):
    fromX, fromY = divmod(fromPos[0],size), divmod(fromPos[1],size)
    toX, toY = divmod(toPos[0],size), divmod(toPos[1],size)
    if fromY < toY: return ShipAction.NORTH
    if fromY > toY: return ShipAction.SOUTH
    if fromX < toX: return ShipAction.EAST
    if fromX > toX: return ShipAction.WEST


def pickRandomShip(me):
    ranship = randint(0,len(me.ships)-1)
    return me.ships[ranship]
        
def pickRandomShipyard(me):
    ranship = randint(0,len(me.shipyards)-1)
    return me.shipyards[ranship]

def getClosestShipyard(ship, me, size): # returns closest shipyard from a ship
    maxdist = -1
    for shipyard in me.shipyards:
        dist = abs(divmod(ship.position[0],size) - divmod(shipyard.position[0],size)) + abs(divmod(ship.position[1],size) - divmod(shipyard.position[1],size))
        if dist > maxdist:
            maxdist = dist
            closest = shipyard
    return closest

def attackShipDir(ship, opponent, size): # attacks most halite carrying ship of an opponent
    maxhal = -1
    for oship in opponent.ships:
        if oship.halite > maxhal:
            maxhal = oship.halite
            target = oship
            
    return getDirTo(ship.position, target.position, size)
    
    
# Directions a ship can move
directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]

# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard
ship_states = {}
attack_ship_id = -1

# Returns the commands we send to our ships and shipyards
def agent(obs, config):
    size = config.size
    board = Board(obs, config)
    me = board.current_player

    # make a shipyard first
    if len(me.shipyards) < 1 and len(me.ships) > 0:
        pickRandomShip(me).next_action = ShipAction.CONVERT
        
        
    # If there are less than 2 ships, use first shipyard to spawn a ship.
    if len(me.ships) < 1 and len(me.shipyards) > 0:   
        pickRandomShipyard(me).next_action = ShipyardAction.SPAWN
        
    if me.halite > 4500: # spawn attack ship
        pickRandomShipyard(me).next_action = ShipyardAction.SPAWN
        attack_ship_id = len(me.ships) - 1
        me.ship[attack_ship_id].next_action = attackShipDir(me.ship[attack_ship_id], board.opponents[0], size)
        
        
    for ship in me.ships:
        if ship.next_action == None:
            
            ### Part 1: Set the ship's state 
            if ship.halite < 100: # If cargo is too low, collect halite
                ship_states[ship.id] = "COLLECT"
                
            if ship.halite > 300: # If cargo gets very big, deposit halite
                ship_states[ship.id] = "DEPOSIT"
                
                
                
            ### Part 2: Use the ship's state to select an action
            if ship_states[ship.id] == "COLLECT":
                # If halite at current location running low, 
                # move to the adjacent square containing the most halite
                if ship.cell.halite < 100:
                    ship.next_action = choice([ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST])
                if ship.cell.halite > 100:
                    ship.next_action = choice([getDirTo(ship.position, me.shipyards[0].position, size), None])
                
                    
            if ship_states[ship.id] == "DEPOSIT":
                # Move towards shipyard to deposit cargo
                direction = getDirTo(ship.position, me.shipyards[0].position, size)
                if direction: ship.next_action = direction
                
    return me.next_actions
