% Solution from Homework 4
function H = estimate_H(xy, XY)
    n = size(XY, 2);
    A = [];
    for i=1:n
        X = XY(1,i);
        Y = XY(2,i);
        x = xy(1,i);
        y = xy(2,i);
        A_i = [X,Y,1, 0,0,0, -X*x, -Y*x, -x ;
               0,0,0, X,Y,1, -X*y, -Y*y, -y ];
        A = [A ; A_i];
    end
    [U,S,V] = svd(A);
    h = V(:,9);
    H = [h(1) h(2) h(3) ;
         h(4) h(5) h(6) ;
         h(7) h(8) h(9) ];
    % Alternatively:
    % H = reshape(h, [3,3])';
end
