# board = [
#     [7,8,0,4,0,0,1,2,0],
#     [6,0,0,0,7,5,0,0,9],
#     [0,0,0,6,0,1,0,7,8],
#     [0,0,7,0,4,0,2,6,0],
#     [0,0,1,0,5,0,9,3,0],
#     [9,0,4,0,6,0,0,0,5],
#     [0,7,0,3,0,0,0,1,2],
#     [1,2,0,0,0,7,4,0,0],
#     [0,4,9,2,0,6,0,0,7]
# ]

def solve(brd):
    find=find_empty(brd)
    if not find:
        return True
    else:
        row , col =find
    for i in range(1,10):
        if valid(brd,i,(row , col)):
            brd[row][col]=i

            if solve(brd):
                return True

            brd[row][col] = 0
    return False
                    

def valid(brd,num,pos):
    for i in range(len(brd[0])):
        if brd[pos[0]][i] == num and pos[1] != i:
            return False
        
    for i in range(len(brd)):
        if brd[i][pos[1]] == num and pos[0] != i:
            return False
            
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3 , box_y*3 + 3):
        for j in range(box_x*3 , box_x*3 + 3):
            if brd[i][j]==num and (i,j) != 0:
                return False
    return True
    
            

    
def print_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i !=0:
            print("- - - - - - - - - - - - - -")

        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print(" | ",end="")

            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ",end="")

def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # row, col

    return None


