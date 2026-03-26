
R = 0.01; % 1 cm 
D = 0.01; % 1 cm 
epsilon = D/R; 

a = 0;
b = 1;

K = @(x,y)(1/pi) * (1./(epsilon^2 + (x-y).^2) + 1./(epsilon^2 + (x+y).^2));

n_values = [10 20 40 80 160];
errors_classic = zeros(size(n_values));
errors_gauss   = zeros(size(n_values));

% Question 5

%Approche linéaire

n = 40; % Nombre de points
x = linspace(a, b, n+1); % Points
h = (b - a)/n; % Ecart entre les points

function [x, u] = solve_linear_classic(n, K, a, b)
    h = (b - a)/n;
    x = linspace(a, b, n+1);
    A = zeros(n+1);

    for i = 1:n+1
        for j = 1:n+1
            phi_j = @(y) max(0, 1 - abs(y - x(j))/h);
            y_min = max(a, x(j) - h);
            y_max = min(b, x(j) + h);
            A(i,j) = integral(@(y) K(x(i), y).*phi_j(y), y_min, y_max);
        end
    end

    u = (eye(n+1) - A) \ ones(n+1,1);
end

% --- Fonction Gauss-Legendre ---
function [x,w] = gauss_legendre(n)
    beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
    T = diag(beta,1) + diag(beta,-1);
    [V,D] = eig(T);
    x = diag(D);
    [x, idx] = sort(x);
    w = 2 * (V(1,idx)').^2;
end

function [x, u] = solve_linear_gauss(n, K, a, b)
    h = (b - a)/n;
    x = linspace(a, b, n+1);
    A = zeros(n+1);

    quad_order = 20;
    [gl_x, gl_w] = gauss_legendre(quad_order);

    for i = 1:n+1
        xi = x(i);

        for j = 1:n+1
            phi_j = @(y) max(0, 1 - abs(y - x(j))/h);
            y_min = max(a, x(j) - h);
            y_max = min(b, x(j) + h);

            if y_max <= y_min
                A(i,j) = 0;
                continue;
            end

            yq = (y_max + y_min)/2 + (y_max - y_min)/2 * gl_x;
            wq = gl_w * (y_max - y_min)/2;

            A(i,j) = sum( K(xi, yq).*phi_j(yq).*wq );
        end
    end

    u = (eye(n+1) - A) \ ones(n+1,1);
end



% Solution de référence très fine (méthode Gauss-Legendre)
n_ref = 800;
[x_ref, u_ref] = solve_linear_gauss(n_ref, K, a, b);

for k = 1:length(n_values)
    n = n_values(k);

    % --- Méthode classique ---
    [x_c, u_c] = solve_linear_classic(n, K, a, b);
    u_ref_interp = interp1(x_ref, u_ref, x_c);
    errors_classic(k) = max(abs(u_c - u_ref_interp'));

    % --- Méthode Gauss-Legendre ---
    [x_g, u_g] = solve_linear_gauss(n, K, a, b);
    u_ref_interp2 = interp1(x_ref, u_ref, x_g);
    errors_gauss(k) = max(abs(u_g - u_ref_interp2'));
end

figure(1);
plot(x_c, u_c);
title('Densité de charge équivalente u(x)');
xlabel('Position normalisée x');
ylabel('u(x)');

figure(2);
plot(x_g, u_g, 'LineWidth', 2);
title('Méthode linéaire avec quadrature de Gauss-Legendre');
xlabel('x');
ylabel('u(x)');
grid on;

% Question 8 ----------

figure(5);
loglog(n_values, errors_classic, 'r-o', 'LineWidth', 2); hold on;
loglog(n_values, errors_gauss, 'b-s', 'LineWidth', 2);
grid on;
xlabel('n');
ylabel('Erreur max ||u - u_n||_\infty');
title('Comparaison des méthodes linéaires');
legend('Classique (integral)', 'Gauss-Legendre');

%---------------

% Question 5

%Approche par interpolation global

function Lj = lagrange_basis(j, x_nodes, y)
    % Évalue le polynôme de Lagrange φ_j(y)
    % x_nodes = points d'interpolation
    n = length(x_nodes);
    Lj = ones(size(y));
    xj = x_nodes(j);

    for k = 1:n
        if k ~= j
            Lj = Lj .* ( (y - x_nodes(k)) / (xj - x_nodes(k)) );
        end
    end
end


function A = build_A_matrix(x_nodes, kernel, a, b, quad_order)
    % quad_order = nombre de point n 
    n = length(x_nodes);
    A = zeros(n);

    % Points et poids de Gauss-Legendre sur [-1,1]
    [yq, wq] = gauss_legendre(quad_order);

    % Rattrapage à  [a,b]
    yq = 0.5*(b-a)*yq + 0.5*(b+a);
    wq = 0.5*(b-a)*wq;

    for i = 1:n
        xi = x_nodes(i);
        for j = 1:n
            Lj = lagrange_basis(j, x_nodes, yq);
            integrand = kernel(xi, yq) .* Lj;
            A(i,j) = sum(integrand .* wq);
        end
    end
end

n2 = 40; % Nombre de points
x2 = linspace(a, b, n2); % Points
A2 = build_A_matrix(x2,K,a,b,n2);
I2 = eye(n2);
d2 = ones(n2, 1); % Terme source d(x) = 1

u2 = (I2 - A2) \ d2;

figure(3);
plot(x2, u2);
title('Densité de charge équivalente u2(x)');
xlabel('Position normalisée x2');
ylabel('u2(x)');


% Question 7

%Erreur

% Valeurs de n demandées par l'énoncé
n_values = [10, 20, 40, 80];
errors = zeros(size(n_values));
h_values = 1 ./ n_values;

% Calcule de la solution de référence
n_ref = 200; %Beaucoup de points pour la référence
[u_ref, x_ref] = solve_condensateur(n_ref, K);

% --- calcul de l'erreur ---
for k = 1:length(n_values)
    n = n_values(k);
    [u_n, x_n] = solve_condensateur(n, K);

    % Interpolation de u_ref sur la grille plus grossière x_n pour comparer
    u_ref_interp = interp1(x_ref, u_ref, x_n);

    % Calcul de l'erreur
    errors(k) = max(abs(u_n - u_ref_interp'));
end

% --- Graphiques ---

% --- Fonction de résolution (Interpolation linéaire) ---
function [u, x] = solve_condensateur(n, K_func)
    x = linspace(0, 1, n+1);
    h = 1/n;
    A = zeros(n+1, n+1);
    for i = 1:n+1
        for j = 1:n+1
            phi_j = @(y) max(0, 1 - abs(y - x(j))/h);
            y_min = max(0, x(j) - h);
            y_max = min(1, x(j) + h);
            A(i,j) = integral(@(y) K_func(x(i), y) .* phi_j(y), y_min, y_max);
        end
    end
    u = (eye(n+1) - A) \ ones(n+1, 1);
end

% Graphique Erreur vs n 

figure(4);
subplot(1,2,1);
loglog(n_values, errors, 'r-o', 'LineWidth', 2);
grid on;
xlabel('Nombre de points n');
ylabel('Erreur e_n');
title('Erreur vs n');

% Graphique Erreur vs h 
subplot(1,2,2);
loglog(h_values, errors, 'b-s', 'LineWidth', 2);
title('Erreur vs h');
xlabel('Pas d''espace h');
ylabel('Erreur e_n');
hold on;
grid on;

% --- Ajout : droite de référence en h^2 ---
ref = errors(1) * (h_values / h_values(1)).^2;
loglog(h_values, ref, 'r--', 'LineWidth', 1.5);
legend('Erreur', 'Référence h^2');

% --- Ajout : calcul de l'ordre ---
p = polyfit(log(h_values), log(errors), 1);
disp(['Ordre de convergence estimé : p = ', num2str(p(1))]);

% Q9
epsilon0 = 8.854e-12; 
R_val = 0.01;        
D_val = 0.01;
epsilon =  D_val/R_val;
integral_u = trapz(x_ref, u_ref); 

C_numerique = 4 * epsilon * R_val * integral_u;

fprintf('Intégrale de u(x) : %.4f\n', integral_u);
fprintf('Capacité calculée (Love) : %.4e F\n', C_numerique);

% Q10
     
S = pi * R_val^2;     

C_classique = (epsilon0 * S) / D_val;

fprintf('Capacité classique (S/D) : %.4e F\n', C_classique);
ecart_relatif = abs(C_numerique - C_classique) / C_numerique * 100;
fprintf('Écart relatif : %.2f %%\n', ecart_relatif);



