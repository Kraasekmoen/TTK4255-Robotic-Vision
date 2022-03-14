classdef Quanser < handle
    properties
        K
        platform_to_camera
        heli_points

        base_to_camera
        hinge_to_camera
        arm_to_camera
        rotors_to_camera
        uv_hat
    end
    methods
        function obj = Quanser()
            obj.K                  = load('../data/K.txt');
            obj.platform_to_camera = load('../data/platform_to_camera.txt');
            obj.heli_points        = load('../data/heli_points.txt')';
        end
        function r = residuals(obj, uv, weights, yaw, pitch, roll)
            % Compute the helicopter coordinate frames
            base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)*rotate_z(yaw);
            hinge_to_base    = translate(0.000, 0.000,  0.325)*rotate_y(pitch);
            arm_to_hinge     = translate(0.000, 0.000, -0.050);
            rotors_to_arm    = translate(0.650, 0.000, -0.030)*rotate_x(roll);
            obj.base_to_camera   = obj.platform_to_camera*base_to_platform;
            obj.hinge_to_camera  = obj.base_to_camera*hinge_to_base;
            obj.arm_to_camera    = obj.hinge_to_camera*arm_to_hinge;
            obj.rotors_to_camera = obj.arm_to_camera*rotors_to_arm;

            % Compute the predicted image location of the markers
            p1 = obj.arm_to_camera*obj.heli_points(:,1:3);
            p2 = obj.rotors_to_camera*obj.heli_points(:,4:7);
            uv_hat = project(obj.K, [p1, p2]);
            obj.uv_hat = uv_hat; % Save for use in draw()

            %
            % TASK: Compute the residual vector
            %
            % Note: The plotting code will not work correctly if you use
            % a different ordering.
            r = zeros(2*7,1); % Placeholder, remove me!
        end
        function draw(obj, uv, weights, image_number)
            fig = figure(1);
            clf(fig);
            I = imread(sprintf('../data/video%04d.jpg', image_number));
            imshow(I);
            hold on;
            valid = weights == 1;
            scatter(uv(1,valid), uv(2,valid), 64, 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'white', 'LineWidth', 1.5);
            scatter(obj.uv_hat(1,:), obj.uv_hat(2,:), 10, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red');
            draw_frame(obj.K, obj.platform_to_camera, 0.05);
            draw_frame(obj.K, obj.base_to_camera, 0.05);
            draw_frame(obj.K, obj.hinge_to_camera, 0.05);
            draw_frame(obj.K, obj.arm_to_camera, 0.05);
            draw_frame(obj.K, obj.rotors_to_camera, 0.05);
            title(sprintf('Reprojected frames and points on image number %d', image_number));
            legend('Observed', 'Predicted');
        end
    end
end
