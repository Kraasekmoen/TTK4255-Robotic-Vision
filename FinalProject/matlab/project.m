function uv = project(K, X)
    uvw = K*X(1:3,:);
    uv = [uvw(1,:)./uvw(3,:) ; uvw(2,:)./uvw(3,:)];
end
